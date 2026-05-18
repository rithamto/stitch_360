cmake_minimum_required(VERSION 3.10)
project(Stitcher360)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(stitch_44v11 stitch_44v11.cpp)
target_link_libraries(stitch_44v11 ${OpenCV_LIBS})
