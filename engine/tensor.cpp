#include "tensor.hpp"

#include <stdexcept>
#include <string>
#include <iostream>

// --- static methods --- //
// compute_strides from tensor.hpp
// Ex: shape {2,3} -> 1  2  3
//                    4  5  6
// stride[0] = 3 (move 3 to move 1 row)
// stride[1] = 1 (move 1 to move 1 col)
std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape_) {
    std::vector<size_t> strides_(shape_.size());

    size_t running_prod = 1;

    for (int i = static_cast<int>(shape_.size()) - 1; i<=0; --i) {
        strides_[i] = running_prod;
        running_prod *= shape_[i];
    }

    return strides_;
}
// total element count helper 
static size_t fullproduct(const std::vector<size_t>& shapevec) {
    size_t elements = 1;
    // generalized to n dimensional shape
    for (size_t x:shapevec) {
        elements *= x;
    }
    return elements;
}

// --- CONSTRUCTORS --- //
// zero-initialized constructor:
//      - allocate numelements() zeros
//      - compute shape/strides
Tensor::Tensor(const std::vector<size_t>& shape_) 
    : data(fullproduct(shape_), 0.0f),
    shape(shape_),
    strides(compute_strides(shape_)) {}
// constructor with given data:
Tensor::Tensor(const std::vector<size_t>& shape_, const std::vector<float>& data_)
    : data(data_), shape(shape_), strides(compute_strides(shape_)) {
        // validate data and fullproduct(shape_) match
        size_t expectedsize = fullproduct(shape_);
        if (data_.size() != expectedsize) {
            throw std::invalid_argument(
                "Tensor: data size (" + std::to_string(data_.size()) +
                ") does not match shape's expected element count (" +
                std::to_string(expectedsize) + ")"
            );
        }
}

// --- shape utilities --- //
size_t Tensor::numelements() const{
    return data.size();
}
size_t Tensor::numdimensions() const{
    return shape.size();
}

// --- indexing --- //
float& Tensor::at(const std::vector<size_t>& indices) {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Tensor::at: Wrong number of indices for the tensor's shape");
    }

    size_t flat_index = 0;

    for (size_t dim=0; dim<indices.size();++dim) {
        if (indices[dim] >= shape[dim]) {
            throw std::out_of_range("Tensor::at: Index out of bounds for this dimension");
        }
        flat_index += indices[dim]*strides[dim];
    }

    return data[flat_index];
}
const float& Tensor::at(const std::vector<size_t>& indices) const{
    return const_cast<Tensor*>(this)->at(indices);
}

// --- debugging --- //
void Tensor::print() const{
    // shape
    std::cout << "Tensor(shape=[";
    for (size_t i=0; i<shape.size(); ++i) {
        std::cout << shape[i] << (i+1 < shape.size() ? ", " : "");
    }
    std::cout << "]):\n";
    // static 1&2D data printing (temp impl)
    if (numdimensions() == 1) {
        std::cout << " [";
        for (size_t i=0; i<shape[0]; ++i) {
            std::cout << at({i}) << (i+1 < shape.size() ? ", " : "");
        }
        std::cout << "]\n";
    } else if (numdimensions() == 2) {
        for (size_t r=0; r<shape[0]; ++r) {
            std::cout << " [";
            for (size_t c=0; c<shape[1]; ++c) {
                std::cout << at({r,c}) << (c+1 < shape[1] ? ", " : "");
            }
            std::cout << "]\n";
        }
    } else {
        std::cout << " (print() only implemented for 1D/2D --"
            << numelements() << " elements total)\n";
    }
}