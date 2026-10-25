#pragma once

#include "tensor.hpp"

Tensor elementwise_add(const Tensor& a, const Tensor& b);
Tensor elementwise_subtract(const Tensor& a, const Tensor& b);
Tensor elementwise_multiply(const Tensor& a, const Tensor& b);
Tensor elementwise_divide(const Tensor& a, const Tensor& b);

Tensor ReLU(const Tensor& a);