#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 创建一个3x3像素的纯绿色图像
    cv::Mat greenImage(3, 3, CV_8UC3, cv::Scalar(0, 255, 0)); // BGR格式，绿色
    
    // 打印Mat数据结构的内容
    std::cout << "Image data in Mat structure (BGR format):" << std::endl;

    // 迭代遍历像素
    for (int i = 0; i < greenImage.rows; ++i) {
        for (int j = 0; j < greenImage.cols; ++j) {
            cv::Vec3b color = greenImage.at<cv::Vec3b>(i, j);
            std::cout << "Pixel at (" << i << ", " << j << "): ["
                      << (int)color[0] << ", "  // 蓝色
                      << (int)color[1] << ", "  // 绿色
                      << (int)color[2] << "]"    // 红色
                      << std::endl;
        }
    }

    return 0;
}