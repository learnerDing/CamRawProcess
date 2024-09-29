#include "Tensor.h"
#include "src/Kernel/Raw10ToRaw8.h"


int main() {
    const int width = 3264;
    const int height = 2464;

    // Allocate memory for RAW10 (float) and RAW8 (float)
    TensorWrapper<float> d_raw10(width, height);
    TensorWrapper<float> d_raw8(width, height);

    // Create host raw10 data
    float* h_raw10 = new float[width * height * 3]; // 3 channels: R, G, B
    for (int i = 0; i < width * height * 3; ++i) {
        h_raw10[i] = static_cast<float>(i % 1024); // Example values for RAW10
    }

    // Copy the RAW10 data to device
    float* d_raw10_host;
    cudaMalloc(&d_raw10_host, width * height * 3 * sizeof(float));
    cudaMemcpy(d_raw10_host, h_raw10, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice);
    d_raw10.data = d_raw10_host;

    // Launch the RAW10 to RAW8 conversion
    launch_raw10_to_raw8_cuda(&d_raw10, &d_raw8, width, height);

    // Copy the result back to host
    float* h_raw8 = new float[width * height * 3];
    cudaMemcpy(h_raw8, d_raw8.data, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    bool success = true;
    for (int i = 0; i < width * height * 3; ++i) {
        float expected = h_raw10[i] / 4.0f; // Expected value after conversion
        if (h_raw8[i] != expected) {
            std::cout << "Mismatch at index " << i << ": expected " << expected << ", got " << h_raw8[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Test passed successfully!" << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }

    // Cleanup
    delete[] h_raw10;
    delete[] h_raw8;
    cudaFree(d_raw10_host);

    return 0;
}