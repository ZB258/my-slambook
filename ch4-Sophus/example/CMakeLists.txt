option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" OFF)
find_package(Pangolin REQUIRED)
if(USE_UBUNTU_20)
    message("You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()

## 原文件中没有这句代码，不会添加引用
find_package(Sophus REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})

## 将Sophus库添加到可执行文件
target_link_libraries(trajectoryError Sophus::Sophus)
