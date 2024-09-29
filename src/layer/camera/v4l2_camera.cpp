// app.cpp
#include <iostream>     // for std::cout, std::cerr
#include <fcntl.h>      // for open
#include <cstring>      // for memset
#include <sys/ioctl.h>  // for ioctl
#include <linux/videodev2.h> // for v4l2

#include <sys/mman.h>
#include <unistd.h>     // for close
#include <cerrno>       // for errno
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

// Constructor
Camera::Camera(const std::string& devicePath) {
    fd = open(devicePath.c_str(), O_RDWR, 0);
    if (fd < 0) {
        std::cerr << "Open " << devicePath << " failed\n";
    }
}

// Destructor
Camera::~Camera() {
    if (fd >= 0) {
        exitCamera();
        close(fd);
    }
}

int Camera::initialize(int width,int height) {
    getCapability();
    getSupportedVideoFormats();
    setVideoFormat(width,height);
    requestBuffers();
    startCamera();
    return 0;
}

void* Camera::captureRawData(unsigned int * size) {
    void *jpeg_ptr = nullptr;
    unsigned int index;

    if (dqBuffer(&jpeg_ptr, size, &index) < 0) {
        std::cerr << "Failed to dequeue buffer\n";
        return nullptr;
    }

    eqBuffer(index);
    return jpeg_ptr;
}

int Camera::getCapability() {
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(struct v4l2_capability));
    int ret = ioctl(fd, VIDIOC_QUERYCAP, &cap); // Get device capabilities
    if (ret < 0) {
        std::cerr << "VIDIOC_QUERYCAP failed (" << ret << ")\n";
        return ret;
    }
    std::cout << "Driver Info:\n"
              << "  Driver Name: " << cap.driver << "\n"
              << "  Card Name: " << cap.card << "\n"
              << "  Bus info: " << cap.bus_info << "\n";
    std::cout << "Device capabilities:\n";
    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
        std::cout << "  support video capture\n";
    }
    if (cap.capabilities & V4L2_CAP_STREAMING) {
        std::cout << "  support streaming i/o\n";
    }
    if (cap.capabilities & V4L2_CAP_READWRITE) {
        std::cout << "  support read i/o\n";
    }
    return ret;
}

int Camera::getSupportedVideoFormats() {
    std::cout << "List device support video format:\n";
    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.index = 0;
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int ret = 0;
    while ((ret = ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)) == 0) {
        fmtdesc.index++;
        std::cout << "  { pixelformat = '" << (char)(fmtdesc.pixelformat & 0xFF)
                  << (char)((fmtdesc.pixelformat >> 8) & 0xFF)
                  << (char)((fmtdesc.pixelformat >> 16) & 0xFF)
                  << (char)((fmtdesc.pixelformat >> 24) & 0xFF)
                  << "', description = '" << fmtdesc.description << "' }\n";
    }
    return ret;
}

int Camera::setVideoFormat(int width, int height) {
    struct v4l2_format fmt({});
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; 
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    int ret = ioctl(fd, VIDIOC_S_FMT, &fmt);
    if (ret < 0) {
        std::cerr << "VIDIOC_S_FMT failed (" << ret << ")\n";
        return ret;
    }

    ret = ioctl(fd, VIDIOC_G_FMT, &fmt);
    if (ret < 0) {
        std::cerr << "VIDIOC_G_FMT failed (" << ret << ")\n";
        return ret;
    }
    
    std::cout << "Stream Format Informations:\n"
              << " type: " << fmt.type << "\n"
              << " width: " << fmt.fmt.pix.width << "\n"
              << " height: " << fmt.fmt.pix.height << "\n"
              << " pixelformat: " << fmt.fmt.pix.pixelformat << "\n"
              << " field: " << fmt.fmt.pix.field << "\n"
              << " bytesperline: " << fmt.fmt.pix.bytesperline << "\n"
              << " sizeimage: " << fmt.fmt.pix.sizeimage << "\n"
              << " colorspace: " << fmt.fmt.pix.colorspace << "\n"
              << " priv: " << fmt.fmt.pix.priv << "\n";
    return ret;
}

int Camera::requestBuffers() {
    struct v4l2_buffer vbuf;

    memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));
    reqbufs.count = REQBUFS_COUNT;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbufs.memory = V4L2_MEMORY_MMAP;

    int ret = ioctl(fd, VIDIOC_REQBUFS, &reqbufs); // Request buffers
    if (ret == -1) {
        std::cerr << "VIDIOC_REQBUFS fail  " << __FUNCTION__ << " " << __LINE__ << "\n";
        return ret;
    }

    for (int i = 0; i < reqbufs.count; i++) {
        memset(&vbuf, 0, sizeof(struct v4l2_buffer));
        vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vbuf.memory = V4L2_MEMORY_MMAP;
        vbuf.index = i;
        ret = ioctl(fd, VIDIOC_QUERYBUF, &vbuf);
        if (ret == -1) {
            std::cerr << "VIDIOC_QUERYBUF fail  " << __FUNCTION__ << " " << __LINE__ << "\n";
            return ret;
        }
        bufs[i].length = vbuf.length;
        bufs[i].start = mmap(NULL, vbuf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, vbuf.m.offset);
        if (bufs[i].start == MAP_FAILED) {
            std::cerr << "mmap fail  " << __FUNCTION__ << " " << __LINE__ << "\n";
            return ret;
        }
        vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vbuf.memory = V4L2_MEMORY_MMAP;
        ret = ioctl(fd, VIDIOC_QBUF, &vbuf); // Queue buffer
        if (ret == -1) {
            std::cerr << "VIDIOC_QBUF err " << __FUNCTION__ << " " << __LINE__ << "\n";
            return ret;
        }
    }
    return ret;
}

int Camera::startCamera() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int ret = ioctl(fd, VIDIOC_STREAMON, &type); // Start camera capture
    if (ret == -1) {
        perror("start_camera");
        return -1;
    }
    std::cout << "camera->start: start capture\n";
    return 0;
}

int Camera::dqBuffer(void **buf, unsigned int *size, unsigned int *index) {
    struct v4l2_buffer vbuf;
    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vbuf.memory = V4L2_MEMORY_MMAP;
    int ret = ioctl(fd, VIDIOC_DQBUF, &vbuf); // Dequeue to take image
    if (ret == -1) {
        perror("camera dqbuf");
        return -1;
    }
    *buf = bufs[vbuf.index].start;
    *size = vbuf.bytesused;
    *index = vbuf.index;    
    return ret;
}

int Camera::eqBuffer(unsigned int index) {
    struct v4l2_buffer vbuf;
    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vbuf.memory = V4L2_MEMORY_MMAP;
    vbuf.index = index;
    int ret = ioctl(fd, VIDIOC_QBUF, &vbuf); // Queue the buffer back
    if (ret == -1) {
        perror("camera->eqbuf");
        return -1;
    }
    return 0;
}

int Camera::stopCamera() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
    if (ret == -1) {
        perror("camera->stop");
        return -1;
    }
    std::cout << "camera->stop: stop capture\n";
    return 0;
}

int Camera::exitCamera() {
    int ret = 0;
    struct v4l2_buffer vbuf;
    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vbuf.memory = V4L2_MEMORY_MMAP;
    
    for (int i = 0; i < reqbufs.count; i++) {
        ret = ioctl(fd, VIDIOC_DQBUF, &vbuf);
        if (ret == -1)
            break;
    }
    
    for (int i = 0; i < reqbufs.count; i++)
        munmap(bufs[i].start, bufs[i].length);
    
    std::cout << "camera->exit: camera exit\n";
    return ret;
}

// 封装调用函数
extern "C" void* CameraGetRaw(const char* devicePath, int width, int height, unsigned int* size) {
    Camera camera(devicePath);
    if (camera.initialize(width, height) < 0) {
        std::cerr << "Failed to initialize camera\n";
        return nullptr;
    }
    
    void* rawData = camera.captureRawData(size);
    
    if (camera.stopCamera() < 0) {
        std::cerr << "Failed to stop camera\n";
    }

    if (camera.exitCamera() < 0) {
        std::cerr << "Failed to exit camera\n";
    }

    return rawData; // 调用方负责后续的释放
}