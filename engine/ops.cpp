#include "ops.hpp"

#include <stdexcept>
#include <string>

static std::string shape_to_str(const std::vector<size_t>& shape) {
    std::string output = "[";
    for (size_t i=0; i<shape.size(); ++i) {
        output += std::to_string(shape[i]) + (i+1 < shape.size() ? ", " : "");
    }
    output += "]";

    return output;
}

Tensor elementwise_add(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument(
            "elementwise_add: Parameter shapes " + 
            shape_to_str(a.shape) + " and " +
            shape_to_str(b.shape) +
            " do not match."
        );
    }

    Tensor result(a.shape);
    for (size_t i=0;i<a.numelements();++i) {
        result.data[i] = a.data[i]+b.data[i];
    }

    return result;
}