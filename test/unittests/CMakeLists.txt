# file(GLOB TEST_SOURCES *.cpp)

add_executable(TestRaw10ToRaw8 Raw10ToRaw8Test.cu)
target_link_libraries(
    TestRaw10ToRaw8 PUBLIC
    -lcudart
    -lcudadevrt
    Raw10ToRaw8)