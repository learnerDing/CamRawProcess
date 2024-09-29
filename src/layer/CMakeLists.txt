add_subdirectory(camera)
add_subdirectory(processing)

add_library(layer "")
target_sources(layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/camera)
target_sources(layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/processing)

target_link_libraries(layer PRIVATE camera)
target_link_libraries(layer PRIVATE processing)