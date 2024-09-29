#pragma once
#include "Tensor.h"
#include "convert.h"
float* generate_gamma_lut(float gamma);
int Gammaprocess(std::vector<float>floatGamma,int rows,int cols);