#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

// 文件路径
string left_file = "./left.png";
string right_file = "./right.png";

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv) {

    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);

    /**************************
     参数解释
    minDisparity:

    这里设置为 0，表示最小视差。这意味着算法将计算从零开始的视差值。
    numDisparities:

    设置为 96，表示要计算的视差范围。这是视差的最大值，视差范围应该是 16 的倍数。这个值影响到算法能处理的深度范围。
    blockSize:

    设置为 9，表示在计算视差时使用的块大小（通常是奇数）。较大的块可以更好地处理纹理不明显的区域，但可能会降低细节。
    P1:

    设置为 8 * 9 * 9，这是平滑项的权重，用于控制视差变化的平滑程度。P1 是小的平滑参数，通常设置为 8 * blockSize^2。
    P2:

    设置为 32 * 9 * 9，这是大平滑项的权重，用于控制视差变化较大的区域的平滑程度。P2 应该比 P1 大，通常设置为 32 * blockSize^2。
    disp12MaxDiff:

    设置为 1，表示左右视差的最大允许差异。这个参数用于控制左右视差一致性检查。
    uniquenessRatio:

    设置为 63，表示唯一性比率。如果一个像素的最佳视差与次佳视差的差异小于该值，则视差被认为是不可靠的。
    speckleWindowSize:

    设置为 10，表示用于去除噪声的窗口大小。
    speckleRange:

    设置为 100，表示视差的范围限制，用于去除小的离散区域。
    disparityMode:

    设置为 32，指定视差的计算模式。
    **************************/
    // Stereo Semi-Global Block Matching
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数

    cv::Mat disparity_sgbm, disparity; // 定义视差图的变量
    sgbm->compute(left, right, disparity_sgbm); // 计算视差图
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f); // 转换为浮点格式，缩放因子为1/16

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud; // 存储点云数据的向量

    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < left.rows; v++) // 遍历每一行
        for (int u = 0; u < left.cols; u++) { // 遍历每一列
            // 检查视差值是否有效
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            // 创建一个四维向量，前三维为 xyz，第四维为颜色
            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); 

            // 根据双目模型计算点的空间位置
            double x = (u - cx) / fx; // 计算 x 坐标,这里是相机坐标系下的归一化坐标
            double y = (v - cy) / fy; // 计算 y 坐标
            double depth = fx * b / (disparity.at<float>(v, u)); // 计算深度
            point[0] = x * depth; // 更新 x 坐标，相机坐标系下的归一化坐标乘深度得到相机坐标系下的实际三维坐标
            point[1] = y * depth; // 更新 y 坐标
            point[2] = depth; // 更新 z 坐标

            pointcloud.push_back(point); // 将点添加到点云中
        }

    // 显示视差图
    cv::imshow("disparity", disparity / 96.0); // 将视差图可视化
    cv::waitKey(0); // 等待按键

    // 画出点云
    showPointCloud(pointcloud); // 调用绘制点云的函数
    return 0; // 返回0，表示程序正常结束
}

//// 可见rgbd/joinMap.cpp中的注释，但参数可能不完全一致，不要直接复制
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}