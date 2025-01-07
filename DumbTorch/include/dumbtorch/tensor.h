#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <stdexcept>

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
            DType dtype = DType::Float32, DeviceType device = DeviceType::CPU)
            : shape_(shape), dtype_(dtype), device_(device) {

            int64_t expected_size = calculateSize();
            if (data.size() != expected_size) {
                throw std::invalid_argument("Data size doesn't match shape");
            }

            // Allocate memory and copy data
            size_t byte_size = expected_size * sizeof(T);
            data_ = std::shared_ptr<void>(
                malloc(byte_size),
                [](void* ptr) { free(ptr); }
            );

            if (device_ == DeviceType::CPU) {
                std::memcpy(data_.get(), data.data(), byte_size);
            }
            else {
                // TODO: Implement CUDA memory allocation and copy
                throw std::runtime_error("CUDA not implemented yet");
            }
        }

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
        T* getData() const {
            return static_cast<T*>(data_.get());
        }

        // Static helpers
        static size_t getSizeOfDType(DType dtype) {
            switch (dtype) {
            case DType::Float32: return sizeof(float);
            case DType::Float64: return sizeof(double);
            case DType::Int32: return sizeof(int32_t);
            case DType::Int64: return sizeof(int64_t);
            default: throw std::runtime_error("Unknown dtype");
            }
        }

    private:
        // Internal data representation
        std::shared_ptr<void> data_;  // Using void* to handle different data types
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