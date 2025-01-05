#pragma once
#include <vector>
#include <memory>

namespace dumbtorch {

    class Tensor;
    class Function;

    class Tensor {
    public:
        
        Tensor(const std::vector<float>& data,
            const std::vector<int>& shape,
            bool requires_grad = false);

        std::vector<float> data_;
        std::vector<int> shape_;

        bool requires_grad_;
        std::vector<float> grad_;
        std::shared_ptr<Function> grad_fn_; // function that created this Tensor

        void backward();
    };

}
