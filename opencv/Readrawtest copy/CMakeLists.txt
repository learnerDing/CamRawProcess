cmake_minimum_required(VERSION 3.10)
project(RawImageViewer)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(RawImageViewer cvreadraw.cpp)
target_link_libraries(RawImageViewer ${OpenCV_LIBS})