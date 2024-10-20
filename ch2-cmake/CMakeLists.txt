# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 2.8)

# 声明一个 cmake 工程
project(HelloSLAM)

# 设置编译模式
# 特性          | Debug 模式               | Release 模式
# --------------|--------------------------|-------------------------------
# 编译优化      | 不进行优化               | 进行多种优化
# 调试信息      | 包含完整调试信息         | 调试信息减少或删除
# 性能          | 较低性能                 | 较高性能
# 文件大小      | 较大                     | 较小
# 使用场景      | 开发和调试阶段           | 产品发布和生产环境
set(CMAKE_BUILD_TYPE "Debug")

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
add_executable(helloSLAM helloSLAM.cpp)

# 添加hello库
## 输出hello.a
## 这是静态库，在程序编译时将库函数写入可执行文件，文件较大，但可以不带dll库分发可执行文件
add_library(hello libHelloSLAM.cpp)

# 共享库
## 输出hello_shared.so
## 这是动态库，程序编译时将动态库的路径写入可执行文件，分发文件时需要把这个动态库一起传输
## 可以使用ldd命令列出可执行文件所需的动态库
add_library(hello_shared SHARED libHelloSLAM.cpp)

# 添加可执行程序调用hello库中函数
add_executable(useHello useHello.cpp)
# 将库文件链接到可执行程序上
target_link_libraries(useHello hello_shared)
