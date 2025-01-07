#include "dumbtorch/ops/basic_ops.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace dumbtorch {
    namespace ops {
        namespace detail {

            template<typename T>
            void add_impl(const T* a, const T* b, T* out, size_t size) {
                for (size_t i = 0; i < size; ++i) {
                    out[i] = a[i] + b[i];
                }
            }

            template<typename T>
            void subtract_impl(const T* a, const T* b, T* out, size_t size) {
                for (size_t i = 0; i < size; ++i) {
                    out[i] = a[i] - b[i];
                }
            }

            template<typename T>
            void multiply_impl(const T* a, const T* b, T* out, size_t size) {
                for (size_t i = 0; i < size; ++i) {
                    out[i] = a[i] * b[i];
                }
            }

            template<typename T>
            void divide_impl(const T* a, const T* b, T* out, size_t size) {
                for (size_t i = 0; i < size; ++i) {
                    if (b[i] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    out[i] = a[i] / b[i];
                }
            }

            // Explicit template instantiations for supported types
            // Addition
            template void add_impl<float>(const float*, const float*, float*, size_t);
            template void add_impl<double>(const double*, const double*, double*, size_t);
            template void add_impl<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t);
            template void add_impl<int64_t>(const int64_t*, const int64_t*, int64_t*, size_t);
            
            // Subtraction
            template void subtract_impl<float>(const float*, const float*, float*, size_t);
            template void subtract_impl<double>(const double*, const double*, double*, size_t);
            template void subtract_impl<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t);
            template void subtract_impl<int64_t>(const int64_t*, const int64_t*, int64_t*, size_t);

            // Multiplication
            template void multiply_impl<float>(const float*, const float*, float*, size_t);
            template void multiply_impl<double>(const double*, const double*, double*, size_t);
            template void multiply_impl<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t);
            template void multiply_impl<int64_t>(const int64_t*, const int64_t*, int64_t*, size_t);

            // Division
            template void divide_impl<float>(const float*, const float*, float*, size_t);
            template void divide_impl<double>(const double*, const double*, double*, size_t);
            template void divide_impl<int32_t>(const int32_t*, const int32_t*, int32_t*, size_t);
            template void divide_impl<int64_t>(const int64_t*, const int64_t*, int64_t*, size_t);

        } // namespace detail

        bool are_shapes_compatible(
            const std::vector<int64_t>& shape1,
            const std::vector<int64_t>& shape2) {

            size_t rank1 = shape1.size();
            size_t rank2 = shape2.size();
            size_t max_rank = std::max(rank1, rank2);

            // Pad shorter shape with ones from the left
            auto get_dim = [max_rank](const std::vector<int64_t>& shape, size_t i) {
                if (i < max_rank - shape.size()) return 1LL;
                return shape[i - (max_rank - shape.size())];
                };

            // Check each dimension
            for (size_t i = 0; i < max_rank; ++i) {
                int64_t dim1 = get_dim(shape1, i);
                int64_t dim2 = get_dim(shape2, i);

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                    return false;
                }
            }

            return true;
        }

        std::vector<int64_t> calculate_broadcast_shape(
            const std::vector<int64_t>& shape1,
            const std::vector<int64_t>& shape2) {

            if (!are_shapes_compatible(shape1, shape2)) {
                throw std::runtime_error("Shapes are not compatible for broadcasting");
            }

            size_t max_rank = std::max(shape1.size(), shape2.size());
            std::vector<int64_t> result(max_rank);

            auto get_dim = [max_rank](const std::vector<int64_t>& shape, size_t i) {
                if (i < max_rank - shape.size()) return 1LL;
                return shape[i - (max_rank - shape.size())];
                };

            for (size_t i = 0; i < max_rank; ++i) {
                int64_t dim1 = get_dim(shape1, i);
                int64_t dim2 = get_dim(shape2, i);
                result[i] = std::max(dim1, dim2);
            }

            return result;
        }

        template<typename T>
        Tensor binary_op(const Tensor& a, const Tensor& b, void (*op)(const T*, const T*, T*, size_t)) {
            if (a.getDType() != b.getDType()) {
                throw std::runtime_error("Tensors must have the same dtype");
            }
            if (a.getDevice() != b.getDevice()) {
                throw std::runtime_error("Tensors must be on the same device");
            }

            auto output_shape = calculate_broadcast_shape(a.getShape(), b.getShape());
            Tensor result(output_shape, a.getDType(), a.getDevice());

            // For now, we only handle non-broadcasted operations
            // TODO: Implement broadcasting
            if (a.getShape() != b.getShape()) {
                throw std::runtime_error("Broadcasting not implemented yet");
            }

            size_t size = std::accumulate(output_shape.begin(), output_shape.end(),
                1LL, std::multiplies<int64_t>());

            op(a.getData<T>(), b.getData<T>(), result.getData<T>(), size);

            return result;
        }

        Tensor add(const Tensor& a, const Tensor& b) {
            switch (a.getDType()) {
            case DType::Float32:
                return binary_op<float>(a, b, detail::add_impl<float>);
            case DType::Float64:
                return binary_op<double>(a, b, detail::add_impl<double>);
            case DType::Int32:
                return binary_op<int32_t>(a, b, detail::add_impl<int32_t>);
            case DType::Int64:
                return binary_op<int64_t>(a, b, detail::add_impl<int64_t>);
            default:
                throw std::runtime_error("Unsupported dtype");
            }
        }

        // Similar changes for subtract, multiply, and divide
        Tensor subtract(const Tensor& a, const Tensor& b) {
            switch (a.getDType()) {
            case DType::Float32:
                return binary_op<float>(a, b, detail::subtract_impl<float>);
            case DType::Float64:
                return binary_op<double>(a, b, detail::subtract_impl<double>);
            case DType::Int32:
                return binary_op<int32_t>(a, b, detail::subtract_impl<int32_t>);
            case DType::Int64:
                return binary_op<int64_t>(a, b, detail::subtract_impl<int64_t>);
            default:
                throw std::runtime_error("Unsupported dtype");
            }
        }

        Tensor multiply(const Tensor& a, const Tensor& b) {
            switch (a.getDType()) {
            case DType::Float32:
                return binary_op<float>(a, b, detail::multiply_impl<float>);
            case DType::Float64:
                return binary_op<double>(a, b, detail::multiply_impl<double>);
            case DType::Int32:
                return binary_op<int32_t>(a, b, detail::multiply_impl<int32_t>);
            case DType::Int64:
                return binary_op<int64_t>(a, b, detail::multiply_impl<int64_t>);
            default:
                throw std::runtime_error("Unsupported dtype");
            }
        }

        Tensor divide(const Tensor& a, const Tensor& b) {
            switch (a.getDType()) {
            case DType::Float32:
                return binary_op<float>(a, b, detail::divide_impl<float>);
            case DType::Float64:
                return binary_op<double>(a, b, detail::divide_impl<double>);
            case DType::Int32:
                return binary_op<int32_t>(a, b, detail::divide_impl<int32_t>);
            case DType::Int64:
                return binary_op<int64_t>(a, b, detail::divide_impl<int64_t>);
            default:
                throw std::runtime_error("Unsupported dtype");
            }
        }

    } // namespace ops
} // namespace dumbtorch