cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(oneLLM LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})


list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
#                      -rdc=true") # not sure the effect of this option, retain it temply

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
message("-- Assign GPU architecture (sm=70 75 80 86)")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)#默认存放静态库的文件夹位置；
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)#默认存放动态库的文件夹位置;
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)#存放可执行软件的目录；
message("PROJECT_SOURCE_DIR"${PROJECT_SOURCE_DIR})#PROJECT_SOURCE_DIR /home/dkernel/scp/CamRawProcess
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)
include_directories(${CMAKE_SOURCE_DIR}/src) #添加各种头文件寻找路径，编译器才找得到
include_directories(${CMAKE_SOURCE_DIR}/src/Kernel)
include_directories(${CMAKE_SOURCE_DIR}/src/layer)
include_directories(${CMAKE_SOURCE_DIR}/src/layer/camera)
include_directories(${CMAKE_SOURCE_DIR}/src/layer/processing)
include_directories(${CMAKE_SOURCE_DIR}/src/utils)

# get_property(includedir DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)#get到includedir信息存放到includedir变量里面
# message(">>> include_dirs="${includedir}) #打印一下目录情况

option (PERF
  "measure model inference performance" #将PERF这个变量赋值为OFF
  OFF
)
option (PRINT_DATA
  "print kernel output to debug"
  OFF
)
option (SAVE_DATA
  "save kernel output to debug"
  OFF
)
if (PERF)
    add_compile_options(-DPERF)#增加全局定义 类似于add_definitions
endif()
if (PRINT_DATA)
    add_compile_options(-DPRINT_DATA)
endif()
if (SAVE_DATA)
    add_compile_options(-DSAVE_DATA)
endif()
#cmake .. -DPRINT_DATA=ON && make
#cmake .. -DPRINT_DATA=ON -DSAVE_DATA=ON && make
#cmake .. -DPERF=ON && make
#cmake .. && make
file(GLOB_RECURSE RawPro_CXX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc)#把cpp文件添加到变量RawPro_CXX_SOURCES
file(GLOB_RECURSE RawPro_CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)#把cu文件添加到变量RawPro_CUDA_SOURCES

add_library(RawProcess STATIC
           ${RawPro_CXX_SOURCES}
           ${RawPro_CUDA_SOURCES}
           )

add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(examples)


add_executable(CamRawProcess main.cpp)
# target_link_libraries(CamRawProcess PRIVATE src)
target_link_libraries(CamRawProcess PUBLIC -lcublas -lcudart -lcudadevrt RawProcess)

#配置opencv
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
message(STATUS "Opencv_include_dir:${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
