add_library(camera v4l2_camera.cpp)
target_link_libraries(camera PRIVATE ${CMAKE_THREAD_LIBS_INIT})