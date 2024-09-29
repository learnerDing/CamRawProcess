#include "v4l2_camera.h"


int main(int argc, char** argv) { 
    Camera camera("/dev/video0");
   if (camera.initialize() < 0) {
        std::cerr << "Failed to initialize camera\n";
        return -1;
    }
    camera.captureImage("1.raw"); // Capture and save an image
     if (camera.stopCamera() < 0) {
        std::cerr << "Failed to stop camera\n";
    }
    
    if (camera.exitCamera() < 0) {
        std::cerr << "Failed to exit camera\n";
    }

    return 0;
}