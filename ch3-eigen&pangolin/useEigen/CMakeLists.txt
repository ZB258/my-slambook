cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# 添加Eigen头文件
## 这是默认的eigen在ubuntu下的安装地址
## 安装命令：sudo apt-get install libeigen3-dev
## 一般使用findpackages命令找eigen库
## find_package(Eigen3 REQUIRED)
## include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("/usr/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)

