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

static void same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
    throw std::invalid_argument(
        "same_shape: Parameter shapes " + 
        shape_to_str(a.shape) + " and " +
        shape_to_str(b.shape) +
        " do not match."
    );
    }
}

Tensor elementwise_add(const Tensor& a, const Tensor& b) {
    same_shape(a,b);

    const size_t numelements = a.numelements();
    Tensor result(a.shape);

    for (size_t i=0;i<numelements;++i) {
        result.data[i] = a.data[i]+b.data[i];
    }

    return result;
}

Tensor elementwise_subtract(const Tensor& a, const Tensor& b) {
    same_shape(a,b);

    const size_t numelements = a.numelements();
    Tensor result(a.shape);

    for (size_t i=0;i<numelements;++i) {
        result.data[i] = a.data[i]-b.data[i];
    }

    return result;
}

Tensor elementwise_multiply(const Tensor& a, const Tensor& b) {
    same_shape(a,b);

    const size_t numelements = a.numelements();
    Tensor result(a.shape);

    for (size_t i=0;i<numelements;++i) {
        result.data[i] = a.data[i]*b.data[i];
    }

    return result;
}

Tensor elementwise_divide(const Tensor& a, const Tensor& b) {
    same_shape(a,b);

    const size_t numelements = a.numelements();
    Tensor result(a.shape);

    for (size_t i=0;i<numelements;++i) {
        result.data[i] = a.data[i]/b.data[i];
    }

    return result;
}

Tensor ReLU(const Tensor& a) {
    Tensor result(a.shape);

    for (size_t i=0;i<a.numelements();++i) {
        if (a.data[i]<0) {
            result.data[i] = 0;
        } 
        else {
            result.data[i] = a.data[i];
        }
    }

    return result;
}