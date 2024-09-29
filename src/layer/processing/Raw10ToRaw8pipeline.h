#include <iostream>
#include <cuda_runtime.h>
#include <cstring> // for memset
#include <cstdint> // for uint16_t
float* Raw10toTensordata(void* raw10image,int rows,int cols );

int Raw10ToRaw8process(void* raw10image,int rows,int cols);