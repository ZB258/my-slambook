cmake_minimum_required(VERSION 3.0)
project(useSophus)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)
## 尝试加入路径
## include_directories( ${Sophus_INCLUDE_DIRS} )

# Eigen
include_directories("/usr/include/eigen3")
add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)

add_subdirectory(example)
