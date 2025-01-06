#pragma once

#include <vector>
#include <memory>
#include <string>

namespace dumbtorch {

    // Forward declarations
    class AutogradContext;

    enum class DeviceType {
        CPU,
        CUDA
    };

    enum class DType {
        Float32,
        Float64,
        Int32,
        Int64
    };

    class Tensor {
    public:
        // Constructors
        Tensor() = default;  // Default constructor for empty tensor

        // Create tensor from raw data
        template<typename T>
        Tensor(const std::vector<T>& data, const std::vector<int64_t>& shape,
            DType dtype = DType::Float32, DeviceType device = DeviceType::CPU);

        // Create tensor with given shape, initialized to zeros
        Tensor(const std::vector<int64_t>& shape, DType dtype = DType::Float32,
            DeviceType device = DeviceType::CPU);

        // Copy constructor and assignment operator (Rule of Five)
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor();

        // Basic operations
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;

        // Utility methods
        std::string toString() const;
        std::vector<int64_t> getShape() const { return shape_; }
        DType getDType() const { return dtype_; }
        DeviceType getDevice() const { return device_; }
        bool requiresGrad() const { return requires_grad_; }
        void setRequiresGrad(bool requires_grad);

        // Data access
        template<typename T>
        T* getData() const;

    private:
        // Internal data representation
        std::shared_ptr<void> data_;
        std::vector<int64_t> shape_;
        DType dtype_ = DType::Float32;
        DeviceType device_ = DeviceType::CPU;

        // Autograd-related members
        bool requires_grad_ = false;
        std::shared_ptr<AutogradContext> grad_ctx_;

        // Helper methods
        int64_t calculateSize() const;
        void allocateMemory();
    };

} // namespace dumbtorch