// 功能：网格投影过滤功能的演示与验证入口。

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "electronic_fly/grid_projection_filter.h"

namespace
{

double readArg(int argc, char** argv, int index, double default_value)
{
    if (index >= argc)
    {
        return default_value;
    }
    return std::atof(argv[index]);
}

}  // namespace

int main(int argc, char** argv)
{
    const double pixel_dx = readArg(argc, argv, 1, 0.0);
    const double pixel_dy = readArg(argc, argv, 2, 0.0);
    const double drone_x = readArg(argc, argv, 3, 0.5);
    const double drone_y = readArg(argc, argv, 4, 0.0);
    const double drone_z = readArg(argc, argv, 5, 1.2);
    const double drone_yaw = readArg(argc, argv, 6, 0.0);
    const double grid_x = readArg(argc, argv, 7, drone_x);
    const double grid_y = readArg(argc, argv, 8, drone_y);

    electronic_fly::CameraProjectionConfig projection;
    projection.intrinsics.image_width = 640;
    projection.intrinsics.image_height = 480;
    projection.intrinsics.fx = 457.0;
    projection.intrinsics.fy = 457.0;
    projection.intrinsics.cx = 320.0;
    projection.intrinsics.cy = 240.0;
    projection.intrinsics.use_distortion = false;
    projection.camera_to_body = electronic_fly::downwardCameraToBody();
    projection.camera_offset_body = electronic_fly::Vec3{0.025, 0.0, 0.0};
    projection.ground_z_world = 0.0;

    electronic_fly::GridRoiConfig roi;
    roi.cell_size_x = 0.5;
    roi.cell_size_y = 0.5;
    roi.margin_x = 0.10;
    roi.margin_y = 0.10;

    electronic_fly::DronePose2D pose;
    pose.x = drone_x;
    pose.y = drone_y;
    pose.z = drone_z;
    pose.yaw = drone_yaw;

    const auto result = electronic_fly::filterCurrentGridByOffset(
        projection,
        roi,
        pose,
        pixel_dx,
        pixel_dy,
        grid_x,
        grid_y);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "valid=" << (result.projection.valid ? "true" : "false")
              << " accepted=" << (result.accepted ? "true" : "false") << "\n";
    std::cout << "world_point=("
              << result.projection.world_point.x << ", "
              << result.projection.world_point.y << ", "
              << result.projection.world_point.z << ")\n";
    std::cout << "grid_delta=("
              << result.roi.dx << ", "
              << result.roi.dy << ") limits=("
              << result.roi.limit_x << ", "
              << result.roi.limit_y << ")\n";
    return result.accepted ? 0 : 1;
}
