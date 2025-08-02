# WebGPU MNIST Classifier

GPU-accelerated neural network for MNIST digit classification using WebGPU and gpu.cpp library.

## Overview

This project implements a 3-layer neural network (784→256→128→10) that runs entirely on the GPU using WebGPU compute shaders. It demonstrates low-level GPU programming for machine learning inference.

## Features

- Full GPU acceleration using WebGPU
- 3-layer feedforward neural network
- Real MNIST dataset support
- Cross-platform GPU support (AMD, NVIDIA, Intel)
- Built from scratch using gpu.cpp

## Prerequisites

- Linux or WSL2 on Windows
- clang++ compiler with C++17 support
- Vulkan drivers
- [gpu.cpp](https://github.com/AnswerDotAI/gpu.cpp) library

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Al13n0231/webgpu-mnist-classifier.git
   cd webgpu-mnist-classifier

2. Download MNIST dataset:
cd data
wget https://github.com/fgnt/mnist/raw/master/t10k-images-idx3-ubyte.gz
wget https://github.com/fgnt/mnist/raw/master/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..

Install gpu.cpp library:

Clone gpu.cpp from https://github.com/AnswerDotAI/gpu.cpp
Build it following their instructions
Update the Makefile paths to point to gpu.cpp location


Build the project:
make all


### Usage
Basic Neural Layer Demo
./nn_layer
Demonstrates a single neural network layer with matrix multiplication and ReLU activation.

Synthetic Data Classification
./mnist_classifier
Tests the network with generated digit patterns.

Real MNIST Classification
./mnist_real
Loads and classifies actual handwritten digits from the MNIST dataset.

### Project Structure
├── nn_layer.cpp          # Single layer implementation
├── mnist_classifier.cpp  # Multi-layer network with synthetic data
├── mnist_real.cpp       # Full implementation with real MNIST data
├── Makefile            # Build configuration
├── data/               # MNIST dataset directory
└── README.md          # This file
Technical Details

### Architecture: 784 (input) → 256 (hidden) → 128 (hidden) → 10 (output)
Activation: ReLU for hidden layers, Softmax for output
Initialization: He initialization for weights
GPU Framework: WebGPU via Dawn implementation
Parallelization: 64-256 threads per workgroup

### Limitations

Inference only - No training/backpropagation implemented
Random weights - Network shows structure but can't actually classify
Fixed architecture - Network structure is hardcoded in WGSL shaders

### Future Improvements

 Load pre-trained weights from PyTorch/TensorFlow
 Implement convolution layers
 Add batch normalization
 Support variable batch sizes
 Implement training (requires autodiff)

### Performance
On AMD Radeon RX 7900 XTX:

Forward pass: <1ms for batch of 4 images
~235,000 operations per image
Theoretical: ~3.5 TFLOPs achievable

### Acknowledgments

gpu.cpp by Answer.AI
MNIST dataset by Yann LeCun et al.
WebGPU specification by W3C