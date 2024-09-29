//此程序用来在cpp代码里面调用cuda kernel
#include <iostream>
#include <cuda_runtime.h>
#include "Gammacuda.h" // 确保正确引用你的 CUDA 函数声明
#include "Gammapipeline.h"
#include <algorithm>
#include <cmath>
// 生成 Gamma 校正查找表
float* generate_gamma_lut(float gamma) {
    // float lut[256]={0};//真是傻逼了，这放在栈空间上面外面调用不了。
    float* lut = new float[256];//放堆上就行了,后面要手动释放 :delete[] host_lut
    for (int i = 0; i < 256; ++i) {
        float normalized = static_cast<float>(i) / 255.0f;
        float corrected = powf(normalized, gamma);//返回normalized的gamma次幂
        lut[i] = static_cast<float>(std::min(255.0f, std::max(0.0f, corrected * 255.0f)));//反归一化
    }

    return lut;
}

// Gamma调用函数
int Gammaprocess(std::vector<float>floatGamma,int rows,int cols) {
    
    int num_pixels = rows * cols;
    int img_size = num_pixels*3*sizeof(float);

     // 在设备上分配内存
    float* d_img_data;
    cudaError_t err = cudaMalloc((void**)&d_img_data, img_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_img_data: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMemcpy(d_img_data, floatGamma.data(), img_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to d_img_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_img_data); 
        return -1;
    }
    
    TensorWrapper<float>* d_img_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            FP32,
                                                                            {3, rows,cols}, 

    // 生成 Gamma 校正查找表
    float gamma = 0.3f;
    float* host_lut = generate_gamma_lut(gamma);
    PrintValues(host_lut,0,255);//打印LUt表
    // 将 LUT 复制到设备
    float* d_gamma_lut;
    err = cudaMalloc((void**)&d_gamma_lut, sizeof(float)*256);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_img_data: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMemcpy(d_gamma_lut, host_lut, 256 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to d_gamma_lut: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_gamma_lut); 
        return -1;
    }
    TensorWrapper<float>* d_lut_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            FP32,
                                                                            {256}, 
                                                                            d_gamma_lut);//256列的行向量
    // 调用 Gamma 矫正
    launchGammaCorrection(d_img_tensor, d_lut_tensor, rows, cols);

    // 将结果从设备拷贝回主机
    cudaMemcpy(floatGamma.data(), d_img_tensor->data, img_size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    delete d_img_tensor;
    delete d_lut_tensor;
    delete[] host_lut;
    return 0;
}