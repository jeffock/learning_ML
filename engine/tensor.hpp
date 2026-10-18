#pragma once

#include <vector>
#include <cstddef>

struct Tensor {
    std::vector<float> data; // tensor in a flat buffer
    std::vector<size_t> shape; // shape of the matrix i.e. nxm
    std::vector<size_t> strides; // flat buffer steps to take per dimension (how long a row is)

    // --- constructors --- // 
    // shape only, filled with zeros
    explicit Tensor(const std::vector<size_t>& shape_);
    // shape and data, filled with data
    // throws if data_.size() doesn't match product of shape dims
    Tensor(const std::vector<size_t>& shape_, const std::vector<float>& data_);

    // --- shape utilities --- //
    // total elements
    size_t numelements() const;
    // number of dimensions
    size_t numdimensions() const;

    // --- indexing --- //
    // ex. tensor.at({1,2}) returns a reference for row 1, col 2
    // reference allows for read/write
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // --- debugging --- //
    void print() const;

private:
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape_);

};