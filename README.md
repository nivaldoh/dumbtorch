# dumbtorch

Minimal PyTorch replica for learning purposes

## Scope

- **Cpp + Python. Mojo project at some point**
- **Only CPU support for now, GPU later**
- **Tensors**
- **Autograd engine**
- **Basic operations**
- **Simple layers**
- **Training loop**
- **Basic tests**

## Roadmap

1. **Set Up C++ Project and Build System**  
   - Configure a CMake (or alternative) build script.  
   - Ensure the environment can build both static and dynamic libraries.

2. **Implement a Basic `Tensor` Class**  
   - Store data (e.g., `std::vector<float>`) and shape information.  
   - Include flags for `requires_grad` and space for gradients.

3. **Add Forward-Only Operations**  
   - Element-wise addition, subtraction, multiplication, division.  
   - Matrix multiplication (2D to start).

4. **Autograd Engine**  
   - Develop a simple “Function” or “Op” mechanism to track forward passes and compute backward gradients.  
   - Store references to parent `Tensor`s in the graph.

5. **Python Bindings (Using pybind11 or Similar)**  
   - Expose C++ `Tensor` and operations to Python.  
   - Allow simple Python scripts/tests to utilize the C++ backend.

6. **Backward Propagation and Gradient Checks**  
   - Implement `backward()` logic for each operation.  
   - Validate correctness with numerical gradient checks.

7. **Higher-Level Layers and Training Loop**  
   - Build a small `Linear` layer and a few activations (ReLU, Sigmoid).  
   - Provide a basic training loop (loss computation, backward pass, parameter update).

8. **Extended Features (Future)**  
   - GPU support (CUDA or alternative).  
   - Explore more complex layers (convolutions, RNNs).  
   - Possibly integrate Mojo for experimentation.