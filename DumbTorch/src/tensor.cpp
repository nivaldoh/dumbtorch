#include "dumbtorch/tensor.h"
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace dumbtorch {

    template<typename T>
    Tensor::Tensor(const std::vector<T>& data, const std::vector<int64_t>& shape,
        DType dtype, DeviceType device)
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

    Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, DeviceType device)
        : shape_(shape), dtype_(dtype), device_(device) {

        allocateMemory();

        // Initialize to zeros
        size_t byte_size = calculateSize() * getSizeOfDType(dtype_);
        if (device_ == DeviceType::CPU) {
            std::memset(data_.get(), 0, byte_size);
        }
        else {
            // TODO: Implement CUDA initialization
            throw std::runtime_error("CUDA not implemented yet");
        }
    }

    // Copy constructor
    Tensor::Tensor(const Tensor& other)
        : shape_(other.shape_),
        dtype_(other.dtype_),
        device_(other.device_),
        requires_grad_(other.requires_grad_) {

        size_t byte_size = calculateSize() * getSizeOfDType(dtype_);
        data_ = std::shared_ptr<void>(
            malloc(byte_size),
            [](void* ptr) { free(ptr); }
        );

        if (device_ == DeviceType::CPU) {
            std::memcpy(data_.get(), other.data_.get(), byte_size);
        }
        else {
            // TODO: Implement CUDA copy
            throw std::runtime_error("CUDA not implemented yet");
        }
    }

    // Move constructor
    Tensor::Tensor(Tensor&& other) noexcept
        : data_(std::move(other.data_)),
        shape_(std::move(other.shape_)),
        dtype_(other.dtype_),
        device_(other.device_),
        requires_grad_(other.requires_grad_),
        grad_ctx_(std::move(other.grad_ctx_)) {

        // Reset other to empty state
        other.dtype_ = DType::Float32;
        other.device_ = DeviceType::CPU;
        other.requires_grad_ = false;
    }

    // Helper methods
    int64_t Tensor::calculateSize() const {
        if (shape_.empty()) return 0;

        int64_t size = 1;
        for (int64_t dim : shape_) {
            size *= dim;
        }
        return size;
    }

    void Tensor::allocateMemory() {
        size_t byte_size = calculateSize() * getSizeOfDType(dtype_);

        if (device_ == DeviceType::CPU) {
            data_ = std::shared_ptr<void>(
                malloc(byte_size),
                [](void* ptr) { free(ptr); }
            );
        }
        else {
            // TODO: Implement CUDA allocation
            throw std::runtime_error("CUDA not implemented yet");
        }
    }

    // Utility functions
    static size_t getSizeOfDType(DType dtype) {
        switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32: return sizeof(int32_t);
        case DType::Int64: return sizeof(int64_t);
        default: throw std::runtime_error("Unknown dtype");
        }
    }

    std::string Tensor::toString() const {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape_[i];
        }
        oss << "], dtype=";

        switch (dtype_) {
        case DType::Float32: oss << "float32"; break;
        case DType::Float64: oss << "float64"; break;
        case DType::Int32: oss << "int32"; break;
        case DType::Int64: oss << "int64"; break;
        }

        oss << ", device=" << (device_ == DeviceType::CPU ? "cpu" : "cuda");
        oss << ", requires_grad=" << (requires_grad_ ? "true" : "false");
        oss << ")";

        return oss.str();
    }

} // namespace dumbtorch