//app.c	
#include <stdio.h>      //for open printf
#include <fcntl.h>      //for Open O_RDWR 文件控制定义
#include <string.h>     //for memset
#include <sys/ioctl.h>  //for ioctl
#include <linux/videodev2.h> //for v4l2
	
#include <sys/mman.h>
#include<unistd.h>     /*Unix 标准函数定义*/
#include<errno.h>      /*错误号定义*/
	

#define VIDEO_WIDTH  320  //采集图像的宽度
#define VIDEO_HEIGHT 240  //采集图像的高度	
	
#define	REQBUFS_COUNT	4	 //缓存区个数
struct v4l2_requestbuffers reqbufs; 	///定义缓冲区
struct cam_buf {
	void *start;
	size_t length;
};
struct cam_buf bufs[REQBUFS_COUNT]; //映射后指向的同一片帧缓冲区
	
	
//查看 摄像头设备的能力	
int get_capability(int fd){
	int ret=0;
	struct v4l2_capability cap;	
	
	memset(&cap, 0, sizeof(struct v4l2_capability)); /*SourceInsight跳转,可看到能力描述
			struct v4l2_capability {
			__u8	driver[16];   驱动名
			__u8	card[32];   设备名
			__u8	bus_info[32]; 总线信息
			__u32   version;  版本
			__u32	capabilities;  设备支持的操作
			__u32	device_caps;
			__u32	reserved[3];
		};
	  */
	ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);  //查看设备能力信息
	if (ret < 0) {
	    printf("VIDIOC_QUERYCAP failed (%d)\n", ret);
	    return ret;
	}
	printf("Driver Info: \n  Driver Name:%s \n  Card Name:%s \n  Bus info:%s \n",cap.driver,cap.card,cap.bus_info);
	printf("Device capabilities: \n"); 	
	if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) { //支持视频捕获(截取一帧图像保存)
	  printf("  support video capture \n");
	}

	if (cap.capabilities & V4L2_CAP_STREAMING) { //支持视频流操作(mmap映射到同一缓冲区队列，后的入队出队 即流入流出，
	  printf("  support streaming i/o\n");
	}

	if(cap.capabilities & V4L2_CAP_READWRITE) { //支持读写（需内核到应用空间拷贝 慢)
	  printf("  support read i/o\n");
	}
	//V4L2_CAP_VIDEO_OVERLAY 支持视频预览（覆盖），指无需帧拷贝，直接存放到显卡的内存，需硬件的DMA支持 -> 实时预览
	//V4L2_CAP_VIDEO_OUTPUT 支持视频输出
	//V4L2_CAP_VBI_CAPTURE  针对老式 模拟显示设备（CRT),很少用
	//V4L2_CAP_RDS_CAPTURE 电台识别	
	return ret;
}
	
//查看 摄像头支持的视频格式
int get_suppurt_video_format(int fd){
	int ret=0;
  printf("List device support video format:  \n");
  struct v4l2_fmtdesc fmtdesc;
  memset(&fmtdesc, 0, sizeof(fmtdesc));
  fmtdesc.index = 0;
  fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  while ((ret = ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)) == 0) //枚举出支持的视频格式
  {
		fmtdesc.index++;
		printf("  { pixelformat = ''%c%c%c%c'', description = ''%s'' }\n",
		          fmtdesc.pixelformat & 0xFF, (fmtdesc.pixelformat >> 8) & 0xFF, (fmtdesc.pixelformat >> 16) & 0xFF, 
		          (fmtdesc.pixelformat >> 24) & 0xFF, fmtdesc.description);
  }		
  return ret;
}	
	
//设置视频格式	
int set_video_format(int fd){	
	int ret=0;
	struct v4l2_format fmt;
	
	memset(&fmt, 0, sizeof(fmt));
	fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width       = VIDEO_WIDTH; 
	fmt.fmt.pix.height      = VIDEO_HEIGHT;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; //注：如果是支持mipg的摄像头最好(白色小摄像头)
                                               //    普通摄像头 用 V4L2_PIX_FMT_YUYV 它比较麻烦，要用到jpeg库
	fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
	ret = ioctl(fd, VIDIOC_S_FMT, &fmt);
	if (ret < 0) {
	    printf("VIDIOC_S_FMT failed (%d)\n", ret);
	    return ret;
	}

	// 获取视频格式
	ret = ioctl(fd, VIDIOC_G_FMT, &fmt);
	if (ret < 0) {
	    printf("VIDIOC_G_FMT failed (%d)\n", ret);
	    return ret;
	}
	// Print Stream Format
	printf("Stream Format Informations:\n");
	printf(" type: %d\n", fmt.type);
	printf(" width: %d\n", fmt.fmt.pix.width);
	printf(" height: %d\n", fmt.fmt.pix.height);
	char fmtstr[8];
	memset(fmtstr, 0, 8);
	memcpy(fmtstr, &fmt.fmt.pix.pixelformat, 4);
	printf(" pixelformat: %s\n", fmtstr);
	printf(" field: %d\n", fmt.fmt.pix.field);
	printf(" bytesperline: %d\n", fmt.fmt.pix.bytesperline);
	printf(" sizeimage: %d\n", fmt.fmt.pix.sizeimage);
	printf(" colorspace: %d\n", fmt.fmt.pix.colorspace);
	printf(" priv: %d\n", fmt.fmt.pix.priv);
	printf(" raw_date: %s\n", fmt.fmt.raw_data);	
	return ret;
}

//申请帧缓冲区
int request_buf(int fd){
	int ret=0;
	int i;
	struct v4l2_buffer vbuf;
	
	memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));
	reqbufs.count	= REQBUFS_COUNT;					//缓存区个数
	reqbufs.type	= V4L2_BUF_TYPE_VIDEO_CAPTURE;
	reqbufs.memory	= V4L2_MEMORY_MMAP;					//设置操作申请缓存的方式:映射 MMAP
	ret = ioctl(fd, VIDIOC_REQBUFS, &reqbufs); //向驱动申请缓存	
	if (ret == -1) {	
		printf("VIDIOC_REQBUFS fail  %s %d\n",__FUNCTION__,__LINE__);
		return ret;
	}
	//循环映射并入队 -> 让内核 和 应用的虚拟地址空间 指向同一片物理内存
	for (i = 0; i < reqbufs.count; i++){
		//真正获取缓存的地址大小
		memset(&vbuf, 0, sizeof(struct v4l2_buffer));
		vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		vbuf.memory = V4L2_MEMORY_MMAP;
		vbuf.index = i;
		ret = ioctl(fd, VIDIOC_QUERYBUF, &vbuf);
		if (ret == -1) {
		  printf("VIDIOC_QUERYBUF fail  %s %d\n",__FUNCTION__,__LINE__);
			return ret;
		}
		//映射缓存到用户空间,通过mmap讲内核的缓存地址映射到用户空间,并切和文件描述符fd相关联
		bufs[i].length = vbuf.length;
		bufs[i].start = mmap(NULL, vbuf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, vbuf.m.offset);
		if (bufs[i].start == MAP_FAILED) {
			printf("mmap fail  %s %d\n",__FUNCTION__,__LINE__);
			return ret;
		}
		//每次映射都会入队,放入缓冲队列
		vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		vbuf.memory = V4L2_MEMORY_MMAP;
		ret = ioctl(fd, VIDIOC_QBUF, &vbuf);
		if (ret == -1) {
			printf("VIDIOC_QBUF err %s %d\n",__FUNCTION__,__LINE__);
			return ret;
		}
	}
	return ret;
}	

//启动采集
int start_camera(int fd)
{
	int ret;
	
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ret = ioctl(fd, VIDIOC_STREAMON, &type); //ioctl控制摄像头开始采集
	if (ret == -1) {
		perror("start_camera");
		return -1;
	}
	fprintf(stdout, "camera->start: start capture\n");
	return 0;
}

//出队取一帧图像
int camera_dqbuf(int fd, void **buf, unsigned int *size, unsigned int *index){
	int ret=0;
	struct v4l2_buffer vbuf;
	vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	vbuf.memory = V4L2_MEMORY_MMAP;
	ret = ioctl(fd, VIDIOC_DQBUF, &vbuf);	//出队,也就是从用户空间取出图片
	if (ret == -1) {
		perror("camera dqbuf ");
		return -1;
	}	
	*buf = bufs[vbuf.index].start;
	*size = vbuf.bytesused;
	*index = vbuf.index;	
	return ret;
}
	
//入队归还帧缓冲	
int camera_eqbuf(int fd, unsigned int index)
{
	int ret;
	struct v4l2_buffer vbuf;

	vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	vbuf.memory = V4L2_MEMORY_MMAP;
	vbuf.index = index;
	ret = ioctl(fd, VIDIOC_QBUF, &vbuf);		//入队
	if (ret == -1) {
		perror("camera->eqbuf");
		return -1;
	}

	return 0;
}
	
//停止视频采集	
int camera_stop(int fd)
{
	int ret;
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
	if (ret == -1) {
		perror("camera->stop");
		return -1;
	}
	fprintf(stdout, "camera->stop: stop capture\n");

	return 0;
}

//退出释放资源
int camera_exit(int fd)
{
	int i;
	int ret=0;
	struct v4l2_buffer vbuf;
	vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	vbuf.memory = V4L2_MEMORY_MMAP;
	
	//出队所有帧缓冲
	for (i = 0; i < reqbufs.count; i++) {
		ret = ioctl(fd, VIDIOC_DQBUF, &vbuf);
		if (ret == -1)
			break;
	}
	
	//取消所有帧缓冲映射
	for (i = 0; i < reqbufs.count; i++)
		munmap(bufs[i].start, bufs[i].length);
	fprintf(stdout, "camera->exit: camera exit\n");
	return ret;
}	
	
int main(int argc, char**argv)
{ 
	int ret;
	char *jpeg_ptr = NULL;
	unsigned int size;
	unsigned int index;

	int fd = open("/dev/video0", O_RDWR, 0);
	if (fd < 0) {
		printf("Open /dev/video0 failed\n");
		return -1;
	} 
	get_capability(fd); //查看 摄像头设备的能力	
	get_suppurt_video_format(fd); //查看摄像头支持的视频格式
	set_video_format(fd); //设置视频格式	
	request_buf(fd); //申请帧缓冲区
	start_camera(fd); //启动采集
	camera_dqbuf(fd, (void **)&jpeg_ptr, &size, &index); //出队取一帧图像

	int pixfd = open("1.jpg", O_WRONLY|O_CREAT, 0666);//打开文件（无则 创建一个空白文件）
	write(pixfd, jpeg_ptr, size); //将一帧图像写入文件

	camera_eqbuf(fd, index); //入队归还帧缓冲 
	camera_stop(fd); //关掉摄像头

	camera_exit(fd);  //退出释放资源
	close(fd);
	return 0;
}	