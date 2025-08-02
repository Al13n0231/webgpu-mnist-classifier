#include <array>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "../../gpu.hpp"

using namespace gpu;

// 3-layer neural network for MNIST digit classification
// Architecture: 784 (input) -> 256 (hidden1) -> 128 (hidden2) -> 10 (output)
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

// ReLU activation function
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
    
    // First layer: 784 -> 256 with ReLU
    var hidden1_sum = vec4<f32>(0.0);
    for (var chunk = 0u; chunk < params.hidden1_size / 4u; chunk++) {
        var layer1_vals = vec4<f32>(0.0);
        
        for (var h = 0u; h < 4u; h++) {
            let h_idx = chunk * 4u + h;
            if (h_idx < params.hidden1_size) {
                var sum = b1[h_idx];
                
                // Compute dot product for this hidden unit
                for (var i = 0u; i < params.input_size; i++) {
                    sum += input[batch_idx * params.input_size + i] * w1[i * params.hidden1_size + h_idx];
                }
                layer1_vals[h] = relu(sum);
            }
        }
        
        // Store intermediate results for next layer
        if (output_idx < params.output_size) {
            // Second layer computation inline
            for (var h = 0u; h < 4u; h++) {
                let h_idx = chunk * 4u + h;
                if (h_idx < params.hidden1_size && layer1_vals[h] > 0.0) {
                    // This hidden unit is active, contribute to layer 2
                    for (var h2 = 0u; h2 < params.hidden2_size; h2++) {
                        hidden1_sum[h2 % 4u] += layer1_vals[h] * w2[h_idx * params.hidden2_size + h2];
                    }
                }
            }
        }
    }
    
    // Complete second layer: 256 -> 128 with ReLU
    var hidden2_vals: array<f32, 128>;
    for (var h2 = 0u; h2 < params.hidden2_size; h2++) {
        var sum = b2[h2];
        
        // More precise computation for second layer
        for (var h1 = 0u; h1 < params.hidden1_size; h1++) {
            var h1_sum = b1[h1];
            for (var i = 0u; i < params.input_size; i++) {
                h1_sum += input[batch_idx * params.input_size + i] * w1[i * params.hidden1_size + h1];
            }
            let h1_activated = relu(h1_sum);
            sum += h1_activated * w2[h1 * params.hidden2_size + h2];
        }
        
        hidden2_vals[h2] = relu(sum);
    }
    
    // Third layer: 128 -> 10 (output logits)
    if (output_idx < params.output_size) {
        var sum = b3[output_idx];
        
        for (var h = 0u; h < params.hidden2_size; h++) {
            sum += hidden2_vals[h] * w3[h * params.output_size + output_idx];
        }
        
        output[batch_idx * params.output_size + output_idx] = sum;
    }
}
)";

// Softmax function for converting logits to probabilities
void softmax(std::vector<float>& logits, size_t batch_size, size_t num_classes) {
    for (size_t b = 0; b < batch_size; ++b) {
        // Find max for numerical stability
        float max_val = *std::max_element(
            logits.begin() + b * num_classes,
            logits.begin() + (b + 1) * num_classes
        );
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            logits[b * num_classes + i] = std::exp(logits[b * num_classes + i] - max_val);
            sum += logits[b * num_classes + i];
        }
        
        // Normalize
        for (size_t i = 0; i < num_classes; ++i) {
            logits[b * num_classes + i] /= sum;
        }
    }
}

// Initialize weights using He initialization (good for ReLU)
void initializeWeights(std::vector<float>& weights, size_t fan_in, size_t fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / fan_in); // He initialization
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (auto& w : weights) {
        w = dist(gen);
    }
}

// Generate synthetic MNIST-like data for testing
void generateTestData(std::vector<float>& data, std::vector<int>& labels, size_t batch_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pixel_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);
    
    // Generate random "handwritten" digits
    for (size_t b = 0; b < batch_size; ++b) {
        labels[b] = label_dist(gen);
        
        // Create a simple pattern for each digit
        for (size_t i = 0; i < 784; ++i) {
            float x = (i % 28) / 28.0f;
            float y = (i / 28) / 28.0f;
            
            // Simple patterns for each digit (very simplified)
            float intensity = 0.0f;
            switch (labels[b]) {
                case 0: // Circle
                    intensity = (std::pow(x - 0.5f, 2) + std::pow(y - 0.5f, 2) < 0.1f) ? 1.0f : 0.0f;
                    break;
                case 1: // Vertical line
                    intensity = (std::abs(x - 0.5f) < 0.1f) ? 1.0f : 0.0f;
                    break;
                default:
                    intensity = pixel_dist(gen) > 0.7f ? 1.0f : 0.0f;
            }
            
            // Add some noise
            intensity += pixel_dist(gen) * 0.2f;
            data[b * 784 + i] = std::min(1.0f, std::max(0.0f, intensity));
        }
    }
}

int main() {
    printf("MNIST Digit Classification with gpu.cpp\n");
    printf("=======================================\n\n");
    
    // Network architecture
    const size_t batch_size = 4;
    const size_t input_size = 784;  // 28x28 images
    const size_t hidden1_size = 256;
    const size_t hidden2_size = 128;
    const size_t output_size = 10;  // 10 digit classes
    
    // Create GPU context
    Context ctx = createContext();
    
    // Prepare data
    std::vector<float> input_data(batch_size * input_size);
    std::vector<int> labels(batch_size);
    generateTestData(input_data, labels, batch_size);
    
    printf("Generated test batch with labels: ");
    for (size_t i = 0; i < batch_size; ++i) {
        printf("%d ", labels[i]);
    }
    printf("\n\n");
    
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
    Tensor input = createTensor(ctx, Shape{batch_size * input_size}, kf32, input_data.data());
    Tensor w1 = createTensor(ctx, Shape{input_size * hidden1_size}, kf32, w1_data.data());
    Tensor b1 = createTensor(ctx, Shape{hidden1_size}, kf32, b1_data.data());
    Tensor w2 = createTensor(ctx, Shape{hidden1_size * hidden2_size}, kf32, w2_data.data());
    Tensor b2 = createTensor(ctx, Shape{hidden2_size}, kf32, b2_data.data());
    Tensor w3 = createTensor(ctx, Shape{hidden2_size * output_size}, kf32, w3_data.data());
    Tensor b3 = createTensor(ctx, Shape{output_size}, kf32, b3_data.data());
    Tensor output = createTensor(ctx, Shape{batch_size * output_size}, kf32);
    
    // Create network parameters buffer
    struct NetworkParams {
        uint32_t batch_size;
        uint32_t input_size;
        uint32_t hidden1_size;
        uint32_t hidden2_size;
        uint32_t output_size;
        uint32_t padding[3]; // Pad to 32 bytes
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
    
    printf("Running forward pass...\n");
    dispatchKernel(ctx, classifier, promise);
    wait(ctx, future);
    
    // Get results
    toCPU(ctx, output, output_data.data(), output_data.size() * sizeof(float));
    
    // Apply softmax to get probabilities
    softmax(output_data, batch_size, output_size);
    
    // Print predictions
    printf("\nPredictions:\n");
    printf("-----------\n");
    for (size_t b = 0; b < batch_size; ++b) {
        printf("Sample %zu (true label: %d):\n", b, labels[b]);
        
        // Find predicted class
        int predicted = 0;
        float max_prob = 0.0f;
        
        for (size_t i = 0; i < output_size; ++i) {
            float prob = output_data[b * output_size + i];
            if (prob > max_prob) {
                max_prob = prob;
                predicted = i;
            }
            printf("  Digit %zu: %.2f%%\n", i, prob * 100.0f);
        }
        
        printf("  Predicted: %d (confidence: %.1f%%)\n\n", predicted, max_prob * 100.0f);
    }
    
    return 0;
}