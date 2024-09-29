#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

int main() {
    // 确定原图片的基本信息
    int rows = 2464;  // 图像的行数
    int cols = 3264;  // 图像的列数
    int channels = 1; // 图像的通道数，灰度图为1

    // 从raw文件中读取数据
    std::string path = "raw8_image.raw";
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return -1;
    }

    // 创建一个矩阵以存储图像数据
    cv::Mat img(rows, cols, CV_8UC1); // CV_8UC1: 8位无符号单通道

    // 读取数据
    file.read(reinterpret_cast<char*>(img.data), rows * cols);
    file.close();

    // 展示图像
    // 创建一个可调整的窗口
    cv::namedWindow("Resizable Window", cv::WINDOW_NORMAL);

    // //设置窗口大小
    // cv::resizeWindow("Resizable Window", 816, 616);
    cv::imshow("Resizable Window", img);
    cv::waitKey();
    cv::destroyAllWindows();

    std::cout << "ok" << std::endl;
    return 0;
}