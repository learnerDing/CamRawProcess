#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

// 使用命名空间以简化代码
using namespace cv;
using namespace std;

// 函数声明
bool readRawImage(const string& filename, Mat& img, int width, int height);
Mat applyAWB(const Mat& img);
Mat applyCCM(const Mat& img, const Mat& ccm);
Mat applyGammaCorrection(const Mat& img, double gamma);

int main(int argc, char** argv)
{
    // 图像文件名
    string filename = "raw8_image.raw";

    // 图像宽度和高度（请根据实际情况修改）
    int width = 3264;   // 例如，3264像素
    int height = 2464;  // 例如，2464像素

    // 读取RAW图像
    Mat rawImg;
    if (!readRawImage(filename, rawImg, width, height))
    {
        cerr << "无法读取RAW图像文件: " << filename << endl;
        return -1;
    }

    // 显示原始灰度图（可选）
    // imshow("Raw Image", rawImg);
    // waitKey(0);

    // 插值算法（Demosaicing）将单通道RAW图像转换为RGB
    Mat rgbImg;
    // 假设Bayer模式为RGGB
    cv::demosaicing(rawImg, rgbImg, cv::COLOR_BayerRG2RGB);

    // 将图像转换为浮点类型以便后续处理
    rgbImg.convertTo(rgbImg, CV_32FC3, 1.0 / 255.0);

    // 自动白平衡（AWB）
    Mat awbImg = applyAWB(rgbImg);

    // 定义色彩矫正矩阵（CCM）
    // 这里使用一个示例矩阵，实际应用中应根据需求调整
    Mat ccm = (Mat_<float>(3,3) << 
                1.0479,  0.0229, -0.0502,
                0.0296,  1.0152, -0.0343,
               -0.0092,  0.0151,  0.7523);
    
    // 应用色彩矫正矩阵
    Mat ccmImg = applyCCM(awbImg, ccm);

    // Gamma校正
    double gamma = 2.2; // 典型Gamma值
    Mat gammaImg = applyGammaCorrection(ccmImg, gamma);

    // 将图像转换回8位无符号整数类型
    Mat finalImg;
    gammaImg.convertTo(finalImg, CV_8UC3, 255.0);

    // 显示最终图像
    imshow("Processed Image", finalImg);
    waitKey(0);

    return 0;
}

// 函数实现

// 读取RAW图像文件
bool readRawImage(const string& filename, Mat& img, int width, int height)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    // 读取所有字节
    vector<unsigned char> buffer(width * height);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    if (!file)
    {
        cerr << "无法读取足够的数据." << endl;
        return false;
    }

    // 创建单通道图像
    img = Mat(height, width, CV_8UC1, buffer.data()).clone();

    file.close();
    return true;
}

// 自动白平衡（灰色世界假设）
Mat applyAWB(const Mat& img)
{
    // 分离RGB通道
    vector<Mat> channels(3);
    split(img, channels);

    // 计算每个通道的平均值
    Scalar avgR = mean(channels[2]); // R
    Scalar avgG = mean(channels[1]); // G
    Scalar avgB = mean(channels[0]); // B

    // 计算增益
    double avgGray = (avgR[0] + avgG[0] + avgB[0]) / 3.0;
    double gainR = avgGray / avgR[0];
    double gainG = avgGray / avgG[0];
    double gainB = avgGray / avgB[0];

    // 应用增益
    channels[2] = channels[2] * gainR;
    channels[1] = channels[1] * gainG;
    channels[0] = channels[0] * gainB;

    // 合并通道
    Mat balancedImg;
    merge(channels, balancedImg);

    // 确保值在[0,1]范围内
    cv::min(balancedImg, 1.0, balancedImg);
    cv::max(balancedImg, 0.0, balancedImg);

    return balancedImg;
}

// 应用色彩矫正矩阵（CCM）
Mat applyCCM(const Mat& img, const Mat& ccm)
{
    Mat correctedImg = Mat::zeros(img.size(), img.type());

    // 分离RGB通道
    vector<Mat> channels(3);
    split(img, channels);

    // 遍历每个像素并应用CCM
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            Vec3f pixel = img.at<Vec3f>(y, x);
            Vec3f corrected;
            corrected[0] = ccm.at<float>(0,0) * pixel[0] + ccm.at<float>(0,1) * pixel[1] + ccm.at<float>(0,2) * pixel[2];
            corrected[1] = ccm.at<float>(1,0) * pixel[0] + ccm.at<float>(1,1) * pixel[1] + ccm.at<float>(1,2) * pixel[2];
            corrected[2] = ccm.at<float>(2,0) * pixel[0] + ccm.at<float>(2,1) * pixel[1] + ccm.at<float>(2,2) * pixel[2];
            correctedImg.at<Vec3f>(y, x) = corrected;
        }
    }

    // 确保值在[0,1]范围内
    cv::min(correctedImg, 1.0, correctedImg);
    cv::max(correctedImg, 0.0, correctedImg);

    return correctedImg;
}

// Gamma校正
Mat applyGammaCorrection(const Mat& img, double gamma)
{
    Mat gammaImg;
    pow(img, 1.0 / gamma, gammaImg);
    return gammaImg;
}