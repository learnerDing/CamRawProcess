//此程序用来在cpp代码里面调用cuda kernel
#include <iostream>
#include <cuda_runtime.h>
#include "AWB.h" // 确保正确引用你的 CUDA 函数声明
#include "AWBpipeline.h"

// 主程序
int AWBprocess(std::vector<float> floatRGB,int rows,int cols)
{
    // 假设你的图像是 RGB 格式，每个像素 3 个通道
    int num_pixels = rows * cols;

    // 在主机上分配内存并初始化图像数据
    int img_size = num_pixels * 3 * sizeof(float); // 每个像素 3 个 float


    // 在设备上分配内存
    float* d_img_data;
    cudaMalloc((void**)&(d_img_data), img_size);

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_img_data, floatRGB.data(), img_size, cudaMemcpyHostToDevice);

    TensorWrapper<float>* d_img_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            FP32,
                                                                            {3, rows,cols}, 
                                                                            d_img_data);
    // 调用水准自动白平衡函数
    launchAWB(d_img_tensor, rows, cols);

    // 将结果从设备拷贝回主机
    cudaMemcpy(floatRGB.data(), d_img_tensor->data, img_size, cudaMemcpyDeviceToHost);

     // 释放分配的设备内存
    cudaFree(d_img_tensor->data);
    delete d_img_tensor;  // 使用 delete 释放 TensorWrapper

    // 释放设备内存
    cudaFree(d_img_data);
    return 0;
}