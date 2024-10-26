#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>


using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    TrajectoryType poses;         // 相机位姿

    ifstream fin("./pose.txt");
    if (!fin) {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {

        /************************************************
        boost::format 是 Boost 库中的一个类，用于格式化字符串。它提供了一种类似于 printf 的方式，但更安全和灵活。
        fmt 是 boost::format 类型的一个实例，表示一个格式化字符串。
        ("./%s/%d.%s") 是格式模板，包含三个占位符：
        %s：表示一个字符串（string），在代码中通常用于表示文件夹名或文件扩展名。
        %d：表示一个整数（integer），在代码中用于表示图像的编号。
        %s：再次表示一个字符串，通常用于表示文件的扩展名（如 png 或 pgm）。
        ./<folder>/<number>.<extension>
        **************************************************/
        boost::format fmt("./%s/%d.%s"); //图像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (auto &d:data)
            fin >> d;
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),//四元数
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;//归一化坐标乘上深度恢复相机坐标系下的三维坐标
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;//相机坐标系下的p，用T*p得到世界坐标系下的三维坐标

                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];   // blue //这里channels固定为3
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
            }
    }

    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud);
    return 0;
}

// 定义一个名为 showPointCloud 的函数，接收一个点云数据的引用
// pointcloud 是一个存储 Vector6d 类型（包含3D坐标和颜色值）的向量
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
    // 检查点云是否为空
    if (pointcloud.empty()) {
        // 如果为空，输出错误信息到标准错误流
        cerr << "Point cloud is empty!" << endl; // 输出到错误流的消息
        // 结束函数执行
        return;
    }

    // 创建一个窗口并绑定 OpenGL 上下文，标题为 "Point Cloud Viewer"，窗口大小为 1024x768 像素
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    // 启用深度测试，以确保绘制时正确处理深度关系
    glEnable(GL_DEPTH_TEST); // GL_DEPTH_TEST：启用深度缓冲
    // 启用混合，以便对透明度进行处理
    glEnable(GL_BLEND); // GL_BLEND：启用颜色混合
    // 设置混合函数，使用源颜色的 alpha 值和目标颜色的反 alpha 值
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 源：透明度，目标：反透明度

    // 创建一个 OpenGL 渲染状态对象，用于设置投影矩阵和视图矩阵
    pangolin::OpenGlRenderState s_cam(
        // 设置投影矩阵，定义视口大小和相机参数
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        /*
            1024, 768: 窗口的宽度和高度
            500, 500: 焦距，影响视图的缩放
            512, 389: 视口的中心位置（通常是窗口的中心）
            0.1: 最近可见面，防止裁剪近处的物体
            1000: 最远可见面，防止裁剪远处的物体
        */
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
        /*
            0, -0.1, -1.8: 相机位置 (x, y, z)
            0, 0, 0: 相机看向的目标点 (x, y, z)
            0.0, -1.0, 0.0: 相机的上方向 (x, y, z)，决定相机的旋转
        */
    );

    // 创建一个显示视图，并设置视图的边界和处理器
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        /*
            0.0, 1.0: 垂直方向上视图的上下边界（占据整个高度）
            pangolin::Attach::Pix(175): 视图的下边界离窗口底部175像素
            1.0: 水平方向上视图的左右边界（占据整个宽度）
            -1024.0f / 768.0f: 视图的宽高比
        */
        .SetHandler(new pangolin::Handler3D(s_cam)); // 使用 3D 处理器处理视图交互

    // 进入主循环，直到用户关闭窗口
    while (pangolin::ShouldQuit() == false) {
        // 清除颜色缓冲和深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        /*
            GL_COLOR_BUFFER_BIT: 清除颜色缓冲，准备绘制新内容
            GL_DEPTH_BUFFER_BIT: 清除深度缓冲，重新计算深度信息
        */

        // 激活当前视图并设置渲染状态
        d_cam.Activate(s_cam); // 激活视图 d_cam，并设置 OpenGL 状态 s_cam
        // 设置清除颜色为白色
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置清除颜色为白色 (RGBA)

        // 设置点的大小为 2 像素
        glPointSize(2); // 设置绘制点的大小为 2 像素
        // 开始绘制点
        glBegin(GL_POINTS); // 开始绘制点（GL_POINTS）
        // 遍历点云中的每一个点 p
        for (auto &p: pointcloud) {
            // 设置当前点的颜色，使用 RGB 值，并将其缩放到 [0, 1] 范围
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            /*
                p[3], p[4], p[5]: RGB 颜色值，取值范围为 0-255
                将颜色值除以 255.0，缩放到 [0, 1] 的范围，适应 OpenGL 的颜色标准
            */
            // 定义当前点的 3D 坐标
            glVertex3d(p[0], p[1], p[2]); // 使用 p[0], p[1], p[2] 定义点的 3D 坐标
        }
        // 结束绘制点
        glEnd(); // 结束点的绘制
        // 刷新窗口以显示渲染内容
        pangolin::FinishFrame(); // 刷新 Pangolin 窗口，显示当前帧
        // 暂停 5 毫秒，控制渲染帧率
        usleep(5000);   // sleep 5 ms
    }
    // 函数结束，返回
    return;
}

