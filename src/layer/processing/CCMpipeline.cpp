//此程序用来在cpp代码里面调用cuda kernel
#include <iostream>
#include <cuda_runtime.h>
#include "CCMcuda.h" // 确保正确引用你的 CUDA 函数声明
#include "CCMpipeline.h"
#define CCMW 3
#define CCMH 3 
// 主程序
int CCMprocess(std::vector<float>floatCCM,int rows,int cols)
{
    // 假设你的图像是 RGB 格式，每个像素 3 个通道
    int num_pixels = rows * cols;
    // 创建色彩校正矩阵
     float h_ccm_data[3][3] = {{2.34403f, 0.00823594f, -0.0795542f},
    {-1.18042f, 1.44385f, 0.0806464f},
    {-0.296824f, -0.556513f, 0.909063f}};
    // 请根据需要替换为自定义的CCM数值

    // 在主机上分配内存并初始化图像数据
    int img_size = num_pixels * 3 * sizeof(float); // 每个像素 3 个 float

    // 在设备上分配内存
    float* d_img_data;
    cudaError_t err = cudaMalloc((void**)&d_img_data, img_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_img_data: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // 在设备上分配内存
    float* d_ccm_data;
    // 注意这里应该分配 9 * sizeof(float) 的内存
    err = cudaMalloc((void**)&d_ccm_data, 9 * sizeof(float)); // 为颜色校正矩阵分配空间
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_ccm_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_img_data); // 释放已经分配的内存
        return -1;
    }
    // 将数据从主机拷贝到设备
    // 将数据从主机拷贝到设备
    err = cudaMemcpy(d_img_data, floatCCM.data(), img_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to d_img_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_img_data); 
        cudaFree(d_ccm_data); 
        return -1;
    }

    err = cudaMemcpy(d_ccm_data, h_ccm_data, 9 * sizeof(float), cudaMemcpyHostToDevice); // 拷贝9个元素
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to d_ccm_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_img_data); 
        cudaFree(d_ccm_data); 
        return -1;
    }

    
    TensorWrapper<float>* d_img_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            FP32,
                                                                            {3, rows,cols}, 
                                                                            d_img_data);
    TensorWrapper<float>* d_ccm_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            FP32,
                                                                            {CCMW,CCMH}, 
                                                                            d_ccm_data);    
    launchCCMGemm(d_img_tensor, d_ccm_tensor,rows, cols);

    // 将结果从设备拷贝回主机
    cudaMemcpy(floatCCM.data(), d_img_tensor->data, img_size, cudaMemcpyDeviceToHost);
    //ccm矩阵不需要再拷贝回来

    // 释放分配的内存

    delete d_img_tensor;
    delete d_ccm_tensor;

    return 0;
}