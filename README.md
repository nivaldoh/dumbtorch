# dumbtorch

Minimal PyTorch replica for learning purposes

## Scope

- **Cpp + Python. Mojo project at some point**
- **CPU + GPU support**
- **Tensors**
- **Autograd engine**
- **Basic operations**
- **Simple layers**
- **Training loop**
- **Basic tests**

# Roadmap

## Phase 1: Core Tensor Implementation (CPU, no parallellism)
1. Tensors

2. Basic ops
   - Element-wise (add, subtract, multiply, divide)
   - Reduction (sum, mean)
   - Shape manipulation (reshape, transpose)
   - Broadcasting

3. Memory Management
   - Smart pointer implementation
   - Memory pool/allocation strategy
   - Reference counting
   - View/shallow copy support

## Phase 2: Autograd Engine
1. Computational Graph
   - Node and edge representation
   - Operation recording
   - Topology management

2. Automatic Differentiation
   - Forward pass recording
   - Backward pass implementation
   - Gradient computation
   - Chain rule application

3. Gradient Management
   - Gradient accumulation
   - Gradient zeroing
   - Detach operations
   - No-grad context

## Phase 3: CUDA Integration
1. Basic CUDA Setup
   - CUDA toolkit integration
   - Device management
   - Stream handling
   - Error checking utilities

2. GPU Tensor Operations
   - Memory allocation on GPU
   - Basic CUDA kernels
   - Element-wise operations
   - Reduction operations

3. CPU-GPU Interface
   - Memory transfer utilities
   - Synchronization primitives
   - Pinned memory support
   - Stream management

## Phase 4: Neural Network Foundation
1. Base Layer Framework
   - Parameter management
   - Forward/backward interface
   - State handling (train/eval modes)

2. Basic Layers
   - Linear layer
   - Activation functions (ReLU, Sigmoid)
   - Loss functions (MSE, CrossEntropy)

3. Training Infrastructure
   - Optimizer base class
   - SGD implementation
   - Parameter update mechanism
   - Training loop utilities

## Phase 5: Python Bindings
1. Core Bindings
   - Tensor class exposure
   - Operation bindings
   - Memory management integration
   - Exception handling

2. Neural Network Interface
   - Layer class bindings
   - Model construction utilities
   - Training loop exposure
   - Optimizer bindings

3. Testing Framework
   - Unit tests for C++ components
   - Python interface tests
   - Integration tests
   - Performance benchmarks

## Extras
1. Add examples

2. Add parallelism for CPU operations


# Project Structure

DumbTorch/
|-- examples/
|   |-- basic_autograd.py
|   |-- mnist.py
|-- include/
|   |-- dumbtorch/
|       |-- nn/
|       |   |-- layer.h
|       |   |-- losses.h
|       |-- ops/
|       |   |-- basic_ops.h
|       |   |-- cuda_ops.h
|       |-- autograd.h
|       |-- tensor.h
|-- python/
|   |-- dumbtorch/
|   |   |-- _C.pyi
|   |   |-- __init__.py
|   |-- setup.py
|-- src/
|   |-- nn/
|   |   |-- layer.cpp
|   |   |-- losses.cpp
|   |-- ops/
|   |   |-- basic_ops.cpp
|   |   |-- cuda_ops.cu
|   |-- autograd.cpp
|   |-- DumbTorch.cpp
|   |-- DumbTorch.h
|   |-- tensor.cpp
|-- tests/
|   |-- cpp/
|   |-- python/
|   |-- CMakeLists.txt
|-- CMakeLists.txt
|-- CMakePresets.json
|-- .gitignore
|-- README.md
