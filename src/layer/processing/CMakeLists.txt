add_library(processing 
    AutoWhiteBalance.cpp 
    CCM.cpp 
    Gamma.cpp 
    Raw10toRaw8.cpp 
    RawScaler.cpp 
    rawscaler.cpp 
    bitmap_image.hpp 
)

target_link_libraries(processing PRIVATE Kernel)