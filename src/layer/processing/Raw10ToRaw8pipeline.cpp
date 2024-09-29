//此程序用来在cpp代码里面调用cuda kernel
#include <iostream>
#include "Raw10ToRaw8.h" // 确保正确引用你的 CUDA 函数声明
#include "Raw10ToRaw8pipeline.h"
//void*的原始raw格式转float*地址
float* Raw10toTensordata(void* raw10image,int rows = 2464,int cols = 3264)
{
    // 计算原始图像的像素数量
    int num_pixels = rows * cols;
    // 为原始数据分配 float 数组
    float* Raw10tensordata = new float[num_pixels];
    
    // 将输入的 raw10image 类型转换为 uint16_t*
    uint16_t* raw10_data = static_cast<uint16_t*>(raw10image);

    // 迭代每个像素并转换成 float
    for (int i = 0; i < num_pixels; ++i) {
        Raw10tensordata[i] = static_cast<float>(raw10_data[i]);
    }

    return Raw10tensordata;
}
// 主程序
int Raw10ToRaw8process(void* raw10image,int rows,int cols)
{   
    float* raw10_h_img_data = Raw10toTensordata(raw10image);
    // 假设你的图像是 RGB 格式，每个像素 3 个通道
    int num_pixels = rows * cols;

    // 在主机上分配内存并初始化图像数据
    int img_size = num_pixels * 3 * sizeof(float); // 每个像素 1个 float
    // float* h_img_data = h_Img_tensor->data;

    // 在设备上分配内存
    float* raw10_d_img_data;
    cudaMalloc((void**)&(raw10_d_img_data), img_size);

    // 将数据从主机拷贝到设备
    cudaMemcpy(raw10_d_img_data, raw10_h_img_data, img_size, cudaMemcpyHostToDevice);
    //GPU上面的TensorWrapper
    TensorWrapper<float>* raw10_d_img_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            FP32,
                                                                            {1, rows,cols}, 
                                                                            raw10_d_img_data);
    // 调用水准自动白平衡函数
    launch_raw10_to_raw8_cuda(raw10_d_img_tensor, rows, cols);

    // 将结果从设备拷贝回主机
    cudaMemcpy(raw10_h_img_data, raw10_d_img_tensor->data, img_size, cudaMemcpyDeviceToHost);

    // 处理结果（例如输出或保存图像）

    // 释放分配的内存
    cudaFree(raw10_d_img_tensor->data);
    cudaFree(raw10_d_img_tensor);
    free(raw10_d_img_data);

    return 0;
}