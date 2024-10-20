cmake_minimum_required(VERSION 2.8)
project(chapter3)

## 原本的会由于c++版本低报错
## set(CMAKE_CXX_FLAGS "-std=c++11")
## 这里设置c++版本为14
set(CMAKE_CXX_STANDARD 14)

## 生成compile_commands.json文件，以供vscode intellisense使用。也可以在cmake时添加如下参数：cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
## mkdir build&&cd build        cmake..     可以看到此时在build文件夹中出现compile_commands.json
## ctrl shift p 并输入 reload window, 左侧文件->打开文件夹,重新打开项目文件夹，也可以使用快捷键ctrl k ctrl o。当然前面的步骤也可以通过关闭并重启vscode解决。不然程序找不到compile_commands.json
## 似乎要使进入build文件夹中点击compile_commands.json才能进行下一步。建议操作一下
## ctrl shift p 并输入 intellisense选择intellisense设置，选择compile_commands.json，得到完美的intellisense

## 也可以直接在c_cpp_properties.json中添加 "compileCommands": "${workspaceFolder}/build/compile_commands.json"
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# 加入各子工程
## cmake会进入这些子目录并执行其中的CMakeLists，文件存储在本CMakeList运行时指定的目录内
add_subdirectory(useEigen)
add_subdirectory(useGeometry)
add_subdirectory(visualizeGeometry)
add_subdirectory(examples)
