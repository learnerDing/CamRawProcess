// Camera.h
#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <fcntl.h>
#include <cstring>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <vector>
#include <string>

#define REQBUFS_COUNT 4    // Number of buffers

struct cam_buf {
    void *start;
    size_t length;
};

class Camera {
public:
    Camera(const std::string& devicePath);
    ~Camera();
    int initialize(int width,int height);
    void* captureRawData(unsigned int* size);
    int stopCamera();
    int exitCamera();
private:
    int fd;
    cam_buf bufs[REQBUFS_COUNT];
    struct v4l2_requestbuffers reqbufs;

    int getCapability();
    int getSupportedVideoFormats();
    int setVideoFormat(int width, int height);
    int requestBuffers();
    int startCamera();
    int dqBuffer(void **buf, unsigned int *size, unsigned int *index);
    int eqBuffer(unsigned int index);
    //int stopCamera();
    //int exitCamera();
};
extern "C" void* CameraGetRaw(const char* devicePath, int width, int height, unsigned int* size);
#endif