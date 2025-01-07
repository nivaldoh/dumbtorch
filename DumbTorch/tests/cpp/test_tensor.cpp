#include <gtest/gtest.h>
#include "dumbtorch/tensor.h"
#include "dumbtorch/ops/basic_ops.h"
#include <cmath>

using namespace dumbtorch;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code if needed
    }
};

// Test tensor creation and properties
TEST_F(TensorTest, CreationAndProperties) {
    // Test creation with data
    std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<int64_t> shape = { 2, 2 };
    Tensor t(data, shape);

    // Check properties
    EXPECT_EQ(t.getShape(), shape);
    EXPECT_EQ(t.getDType(), DType::Float32);
    EXPECT_EQ(t.getDevice(), DeviceType::CPU);

    // Test empty creation with zeros
    Tensor zeros({ 2, 3 }, DType::Float32);
    auto zeros_data = zeros.getData<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(zeros_data[i], 0.0f);
    }
}

// Test memory management
TEST_F(TensorTest, MemoryManagement) {
    std::vector<float> data = { 1.0f, 2.0f, 3.0f };
    std::vector<int64_t> shape = { 3 };
    Tensor original(data, shape);

    // Test copy constructor
    Tensor copy = original;
    auto orig_data = original.getData<float>();
    auto copy_data = copy.getData<float>();
    EXPECT_NE(orig_data, copy_data);  // Different memory addresses
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(orig_data[i], copy_data[i]);  // Same values
    }

    // Test move constructor
    Tensor moved = std::move(copy);
    auto moved_data = moved.getData<float>();
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(moved_data[i], data[i]);  // Should have original's data
    }
}

// Test basic operations
TEST_F(TensorTest, BasicOperations) {
    std::vector<float> data1 = { 1.0f, 2.0f, 3.0f };
    std::vector<float> data2 = { 4.0f, 5.0f, 6.0f };
    std::vector<int64_t> shape = { 3 };

    Tensor a(data1, shape);
    Tensor b(data2, shape);

    // Test addition
    Tensor c = a + b;
    auto c_data = c.getData<float>();
    EXPECT_EQ(c_data[0], 5.0f);
    EXPECT_EQ(c_data[1], 7.0f);
    EXPECT_EQ(c_data[2], 9.0f);

    // Test multiplication
    Tensor d = a * b;
    auto d_data = d.getData<float>();
    EXPECT_EQ(d_data[0], 4.0f);
    EXPECT_EQ(d_data[1], 10.0f);
    EXPECT_EQ(d_data[2], 18.0f);
}

// Test error handling
TEST_F(TensorTest, ErrorHandling) {
    std::vector<float> data1 = { 1.0f, 2.0f };
    std::vector<float> data2 = { 1.0f, 2.0f, 3.0f };
    std::vector<int64_t> shape1 = { 2 };
    std::vector<int64_t> shape2 = { 3 };

    Tensor a(data1, shape1);
    Tensor b(data2, shape2);

    // Test shape mismatch
    EXPECT_THROW(a + b, std::runtime_error);

    // Test division by zero
    std::vector<float> zeros = { 0.0f, 0.0f };
    Tensor zero(zeros, shape1);
    EXPECT_THROW(a / zero, std::runtime_error);
}

// Test different data types
TEST_F(TensorTest, DataTypes) {
    // Int32 tests
    std::vector<int32_t> int_data = { 1, 2, 3 };
    std::vector<int64_t> shape = { 3 };
    Tensor int_tensor(int_data, shape, DType::Int32);
    EXPECT_EQ(int_tensor.getDType(), DType::Int32);
    auto int_tensor_data = int_tensor.getData<int32_t>();
    EXPECT_EQ(int_tensor_data[0], 1);

    // Float64 tests
    std::vector<double> double_data = { 1.0, 2.0, 3.0 };
    Tensor double_tensor(double_data, shape, DType::Float64);
    EXPECT_EQ(double_tensor.getDType(), DType::Float64);
    auto double_tensor_data = double_tensor.getData<double>();
    EXPECT_EQ(double_tensor_data[0], 1.0);
}

// Test type compatibility
TEST_F(TensorTest, TypeCompatibility) {
    std::vector<float> float_data = { 1.0f, 2.0f };
    std::vector<int32_t> int_data = { 1, 2 };
    std::vector<int64_t> shape = { 2 };

    Tensor float_tensor(float_data, shape, DType::Float32);
    Tensor int_tensor(int_data, shape, DType::Int32);

    // Operations between different types should throw
    EXPECT_THROW(float_tensor + int_tensor, std::runtime_error);
}