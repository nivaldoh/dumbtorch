#pragma once

#include "dumbtorch/tensor.h"
#include <memory>

namespace dumbtorch {
    namespace ops {

        // Forward declarations of internal implementation functions
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

            constexpr size_t PARALLEL_THRESHOLD = 1024;  // Tune this value based on benchmarks
        } // namespace detail

        // Main operation functions
        Tensor add(const Tensor& a, const Tensor& b);
        Tensor subtract(const Tensor& a, const Tensor& b);
        Tensor multiply(const Tensor& a, const Tensor& b);
        Tensor divide(const Tensor& a, const Tensor& b);

        // Broadcasting helper
        std::vector<int64_t> calculate_broadcast_shape(
            const std::vector<int64_t>& shape1,
            const std::vector<int64_t>& shape2);

        // Shape validation
        bool are_shapes_compatible(
            const std::vector<int64_t>& shape1,
            const std::vector<int64_t>& shape2);

    } // namespace ops
} // namespace dumbtorch