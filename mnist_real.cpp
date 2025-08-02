#include <array>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "../../gpu.hpp"

using namespace gpu;

// Same network architecture as before
static const char* kMNISTNetwork = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;       // [batch_size, 784]
@group(0) @binding(1) var<storage, read> w1: array<f32>;          // [784, 256]
@group(0) @binding(2) var<storage, read> b1: array<f32>;          // [256]
@group(0) @binding(3) var<storage, read> w2: array<f32>;          // [256, 128]
@group(0) @binding(4) var<storage, read> b2: array<f32>;          // [128]
@group(0) @binding(5) var<storage, read> w3: array<f32>;          // [128, 10]
@group(0) @binding(6) var<storage, read> b3: array<f32>;          // [10]
@group(0) @binding(7) var<storage, read_write> output: array<f32>; // [batch_size, 10]

struct NetworkParams {
    batch_size: u32,
    input_size: u32,
    hidden1_size: u32,
    hidden2_size: u32,
    output_size: u32,
}

@group(0) @binding(8) var<uniform> params: NetworkParams;

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let batch_idx = thread_id / params.output_size;
    let output_idx = thread_id % params.output_size;
    
    if (batch_idx >= params.batch_size) {
        return;
    }
    
    // Layer 1: 784 -> 256 with ReLU
    var hidden1: array<f32, 256>;
    for (var h = 0u; h < params.hidden1_size; h++) {
        var sum = b1[h];
        for (var i = 0u; i < params.input_size; i++) {
            sum += input[batch_idx * params.input_size + i] * w1[i * params.hidden1_size + h];
        }
        hidden1[h] = relu(sum);
    }
    
    // Layer 2: 256 -> 128 with ReLU
    var hidden2: array<f32, 128>;
    for (var h = 0u; h < params.hidden2_size; h++) {
        var sum = b2[h];
        for (var i = 0u; i < params.hidden1_size; i++) {
            sum += hidden1[i] * w2[i * params.hidden2_size + h];
        }
        hidden2[h] = relu(sum);
    }
    
    // Layer 3: 128 -> 10 (output logits)
    if (output_idx < params.output_size) {
        var sum = b3[output_idx];
        for (var h = 0u; h < params.hidden2_size; h++) {
            sum += hidden2[h] * w3[h * params.output_size + output_idx];
        }
        output[batch_idx * params.output_size + output_idx] = sum;
    }
}
)";

// MNIST file format readers
uint32_t readUint32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), 4);
    // MNIST files are big-endian, so swap bytes if needed
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

// Load MNIST images
bool loadMNISTImages(const std::string& filename, std::vector<float>& images, 
                     int& num_images, int& image_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    // Read header
    uint32_t magic = readUint32(file);
    if (magic != 0x00000803) {
        std::cerr << "Invalid MNIST image file!" << std::endl;
        return false;
    }
    
    num_images = readUint32(file);
    int rows = readUint32(file);
    int cols = readUint32(file);
    image_size = rows * cols;
    
    // Read all images
    images.resize(num_images * image_size);
    for (int i = 0; i < num_images * image_size; ++i) {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        images[i] = static_cast<float>(pixel) / 255.0f; // Normalize to [0, 1]
    }
    
    return true;
}

// Load MNIST labels
bool loadMNISTLabels(const std::string& filename, std::vector<int>& labels, int& num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    // Read header
    uint32_t magic = readUint32(file);
    if (magic != 0x00000801) {
        std::cerr << "Invalid MNIST label file!" << std::endl;
        return false;
    }
    
    num_labels = readUint32(file);
    labels.resize(num_labels);
    
    // Read all labels
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
    
    return true;
}

// Display MNIST digit as ASCII art
void displayDigit(const std::vector<float>& images, int index) {
    const int width = 28;
    const int height = 28;
    
    std::cout << "\n+------------------------------+\n";
    
    for (int y = 0; y < height; y += 2) { // Skip every other row for compact display
        std::cout << "| ";
        for (int x = 0; x < width; ++x) {
            float pixel = images[index * width * height + y * width + x];
            
            // Convert pixel intensity to ASCII character
            if (pixel > 0.8f) std::cout << "#";
            else if (pixel > 0.6f) std::cout << "*";
            else if (pixel > 0.4f) std::cout << "+";
            else if (pixel > 0.2f) std::cout << ".";
            else std::cout << " ";
        }
        std::cout << " |\n";
    }
    
    std::cout << "+------------------------------+\n";
}

// Output layer
// Softmax function
void softmax(std::vector<float>& logits, size_t batch_size, size_t num_classes) {
    for (size_t b = 0; b < batch_size; ++b) {
        float max_val = *std::max_element(
            logits.begin() + b * num_classes,
            logits.begin() + (b + 1) * num_classes
        );
        
        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            logits[b * num_classes + i] = std::exp(logits[b * num_classes + i] - max_val);
            sum += logits[b * num_classes + i];
        }
        
        for (size_t i = 0; i < num_classes; ++i) {
            logits[b * num_classes + i] /= sum;
        }
    }
}

// Initialize weights
void initializeWeights(std::vector<float>& weights, size_t fan_in, size_t fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (auto& w : weights) {
        w = dist(gen);
    }
}

int main() {
    printf("MNIST Real Data Classification with gpu.cpp\n");
    printf("===========================================\n\n");
    
    // Load MNIST test data
    std::vector<float> all_images;
    std::vector<int> all_labels;
    int num_images, num_labels, image_size;
    
    printf("Loading MNIST test data...\n");
    if (!loadMNISTImages("data/t10k-images-idx3-ubyte", all_images, num_images, image_size) ||
        !loadMNISTLabels("data/t10k-labels-idx1-ubyte", all_labels, num_labels)) {
        printf("Error: Could not load MNIST data files.\n");
        printf("Make sure you have downloaded the MNIST test files to the 'data' directory.\n");
        return 1;
    }
    
    printf("Loaded %d test images of size %d\n\n", num_images, image_size);
    
    // Network parameters
    const size_t batch_size = 4;
    const size_t input_size = 784;
    const size_t hidden1_size = 256;
    const size_t hidden2_size = 128;
    const size_t output_size = 10;
    
    // Select random batch of images
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, num_images - 1);
    
    std::vector<float> batch_images(batch_size * input_size);
    std::vector<int> batch_labels(batch_size);
    
    printf("Selected batch:\n");
    for (size_t i = 0; i < batch_size; ++i) {
        int idx = dist(gen);
        batch_labels[i] = all_labels[idx];
        
        // Copy image data
        std::copy(all_images.begin() + idx * image_size,
                  all_images.begin() + (idx + 1) * image_size,
                  batch_images.begin() + i * image_size);
        
        printf("\nImage %zu - Label: %d\n", i, batch_labels[i]);
        displayDigit(all_images, idx);
    }
    
    // Create GPU context
    Context ctx = createContext();
    
    // Initialize network weights
    std::vector<float> w1_data(input_size * hidden1_size);
    std::vector<float> b1_data(hidden1_size, 0.0f);
    std::vector<float> w2_data(hidden1_size * hidden2_size);
    std::vector<float> b2_data(hidden2_size, 0.0f);
    std::vector<float> w3_data(hidden2_size * output_size);
    std::vector<float> b3_data(output_size, 0.0f);
    std::vector<float> output_data(batch_size * output_size);
    
    initializeWeights(w1_data, input_size, hidden1_size);
    initializeWeights(w2_data, hidden1_size, hidden2_size);
    initializeWeights(w3_data, hidden2_size, output_size);
    
    // Create GPU tensors
    Tensor input = createTensor(ctx, Shape{batch_size * input_size}, kf32, batch_images.data());
    Tensor w1 = createTensor(ctx, Shape{input_size * hidden1_size}, kf32, w1_data.data());
    Tensor b1 = createTensor(ctx, Shape{hidden1_size}, kf32, b1_data.data());
    Tensor w2 = createTensor(ctx, Shape{hidden1_size * hidden2_size}, kf32, w2_data.data());
    Tensor b2 = createTensor(ctx, Shape{hidden2_size}, kf32, b2_data.data());
    Tensor w3 = createTensor(ctx, Shape{hidden2_size * output_size}, kf32, w3_data.data());
    Tensor b3 = createTensor(ctx, Shape{output_size}, kf32, b3_data.data());
    Tensor output = createTensor(ctx, Shape{batch_size * output_size}, kf32);
    
    // Network parameters
    struct NetworkParams {
        uint32_t batch_size;
        uint32_t input_size;
        uint32_t hidden1_size;
        uint32_t hidden2_size;
        uint32_t output_size;
        uint32_t padding[3];
    };
    
    NetworkParams params{
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(input_size),
        static_cast<uint32_t>(hidden1_size),
        static_cast<uint32_t>(hidden2_size),
        static_cast<uint32_t>(output_size),
        {0, 0, 0}
    };
    
    float params_data[8];
    memcpy(params_data, &params, sizeof(params));
    Tensor params_buffer = createTensor(ctx, Shape{8}, kf32, params_data);
    
    // Create and dispatch kernel
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    
    Kernel classifier = createKernel(ctx,
        {kMNISTNetwork, 64, kf32},
        Bindings{input, w1, b1, w2, b2, w3, b3, output, params_buffer},
        {cdiv(batch_size * output_size, 64), 1, 1}
    );
    
    printf("\n\nRunning forward pass on GPU...\n");
    dispatchKernel(ctx, classifier, promise);
    wait(ctx, future);
    
    // Get results
    toCPU(ctx, output, output_data.data(), output_data.size() * sizeof(float));
    
    // Apply softmax
    softmax(output_data, batch_size, output_size);
    
    // Print predictions
    printf("\nPredictions (with random weights):\n");
    printf("==================================\n");
    for (size_t b = 0; b < batch_size; ++b) {
        printf("\nImage %zu (true label: %d):\n", b, batch_labels[b]);
        
        int predicted = 0;
        float max_prob = 0.0f;
        
        for (size_t i = 0; i < output_size; ++i) {
            float prob = output_data[b * output_size + i];
            if (prob > max_prob) {
                max_prob = prob;
                predicted = i;
            }
            printf("  Digit %zu: %5.2f%%", i, prob * 100.0f);
            if (static_cast<int>(i) == batch_labels[b]) printf(" <-- True label");
            printf("\n");
        }
        
        printf("  Predicted: %d (confidence: %.1f%%) ", predicted, max_prob * 100.0f);
        if (predicted == batch_labels[b]) {
            printf("✓ CORRECT (by luck!)");
        } else {
            printf("✗ WRONG");
        }
        printf("\n");
    }
    
    printf("\nNote: Predictions are random because weights are not trained!\n");
    printf("With proper training, this network could achieve ~98%% accuracy.\n");
    
    return 0;
}
