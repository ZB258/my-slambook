cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release")

## 原本的会由于c++版本低报错
## set(CMAKE_CXX_FLAGS "-std=c++11 -02")
## 这里设置c++版本为14
set(CMAKE_CXX_STANDARD 14)

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)
