#include <array>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include "../../gpu.hpp"

using namespace gpu;

// WGSL shader code for matrix multiplication + ReLU activation
static const char* kMatMulReLU = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;      // [batch_size, input_size]
@group(0) @binding(1) var<storage, read> weights: array<f32>;    // [input_size, output_size]
@group(0) @binding(2) var<storage, read> bias: array<f32>;       // [output_size]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [batch_size, output_size]

struct Dimensions {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
}

@group(0) @binding(4) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;  // batch index
    let col = global_id.x;  // output neuron index
    
    // Check bounds
    if (row >= dims.batch_size || col >= dims.output_size) {
        return;
    }
    
    // Matrix Multiplication Compute dot product: input[row] @ weights[:, col]
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.input_size; i++) {
        let input_idx = row * dims.input_size + i;
        let weight_idx = i * dims.output_size + col;
        sum += input[input_idx] * weights[weight_idx];
    }
    
    // Add bias
    sum += bias[col];
    
    // Apply ReLU activation
    let activated = max(0.0, sum);
    
    // Store result
    let output_idx = row * dims.output_size + col;
    output[output_idx] = activated;
}
)";

// Initialize weights with random values
void initializeWeights(std::vector<float>& weights, float scale = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (auto& w : weights) {
        w = dist(gen);
    }
}

int main() {
    printf("Neural Network Layer with gpu.cpp\n");
    printf("=================================\n\n");
    
    // Network dimensions
    // Total multiplications done per sample:~235,000
    // Batch of 4: ~940,000
    const size_t batch_size = 4;
    const size_t input_size = 784;  // 28x28 image flattened
    const size_t output_size = 128; // hidden layer size
    
    // Create GPU context
    Context ctx = createContext();
    
    // Prepare data
    std::vector<float> input_data(batch_size * input_size);
    std::vector<float> weights_data(input_size * output_size);
    std::vector<float> bias_data(output_size, 0.0f);
    std::vector<float> output_data(batch_size * output_size);
    
    // Initialize with dummy data (normally this would be actual data)
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 255) / 255.0f; // Normalize to [0, 1]
    }
    
    // Initialize weights using Xavier initialization
    initializeWeights(weights_data, sqrtf(2.0f / input_size));
    
    // Create GPU tensors
    Tensor input = createTensor(ctx, Shape{batch_size * input_size}, kf32, input_data.data());
    Tensor weights = createTensor(ctx, Shape{input_size * output_size}, kf32, weights_data.data());
    Tensor bias = createTensor(ctx, Shape{output_size}, kf32, bias_data.data());
    Tensor output = createTensor(ctx, Shape{batch_size * output_size}, kf32);
    
    // Create dimensions uniform buffer
    struct Dimensions {
        uint32_t batch_size;
        uint32_t input_size;
        uint32_t output_size;
        uint32_t padding; // Add padding to make it 16 bytes
    };
    Dimensions dims{static_cast<uint32_t>(batch_size), 
                   static_cast<uint32_t>(input_size), 
                   static_cast<uint32_t>(output_size), 
                   0};
    
    // Create a float array from the dimensions struct
    float dims_data[4] = {
        static_cast<float>(dims.batch_size),
        static_cast<float>(dims.input_size),
        static_cast<float>(dims.output_size),
        0.0f
    };
    Tensor dims_buffer = createTensor(ctx, Shape{4}, kf32, dims_data);
    // Create kernel
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    
    Kernel layer = createKernel(ctx, 
        {kMatMulReLU, /* workgroup size */ 256, kf32},
        Bindings{input, weights, bias, output, dims_buffer},
        /* dispatch size */ {cdiv(output_size, 16), cdiv(batch_size, 16), 1}
    );
    
    // Dispatch the computation
    printf("Running forward pass...\n");
    dispatchKernel(ctx, layer, promise);
    
    // Wait for completion
    wait(ctx, future);
    
    // Get results back to CPU
    toCPU(ctx, output, output_data.data(), output_data.size() * sizeof(float));
    
    // Print some results
    printf("\nResults (first 10 neurons of first sample):\n");
    for (size_t i = 0; i < 10; ++i) {
        printf("  Neuron %zu: %.4f\n", i, output_data[i]);
    }
    
    // Compute some statistics
    float max_activation = 0.0f;
    float mean_activation = 0.0f;
    size_t active_neurons = 0;
    
    for (const auto& val : output_data) {
        max_activation = std::max(max_activation, val);
        mean_activation += val;
        if (val > 0.0f) active_neurons++;
    }
    mean_activation /= output_data.size();
    
    printf("\nLayer Statistics:\n");
    printf("  Max activation: %.4f\n", max_activation);
    printf("  Mean activation: %.4f\n", mean_activation);
    printf("  Active neurons: %zu / %zu (%.1f%%)\n", 
           active_neurons, output_data.size(), 
           100.0f * active_neurons / output_data.size());
    
    return 0;
}