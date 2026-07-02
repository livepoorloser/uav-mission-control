#pragma once

#include <array>
#include <cmath>
#include <limits>

namespace electronic_fly
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Matrix3
{
    // Row-major: body_or_world_vector = R * source_vector.
    std::array<double, 9> m{{1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0}};
};

struct CameraIntrinsics
{
    int image_width = 640;
    int image_height = 480;
    double fx = 457.0;
    double fy = 457.0;
    double cx = 320.0;
    double cy = 240.0;
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double k3 = 0.0;
    bool use_distortion = false;
};

struct CameraProjectionConfig
{
    CameraIntrinsics intrinsics;

    // Camera optical frame uses OpenCV convention: +x right, +y image down, +z optical axis.
    // Body frame convention here: +x forward, +y left, +z up.
    Matrix3 camera_to_body = Matrix3{};

    // Camera center relative to the flight controller/body origin, in body frame, meters.
    Vec3 camera_offset_body;

    // Ground plane z in the same world/local frame as the drone pose.
    double ground_z_world = 0.0;
    double min_ray_down_z = 1e-3;
};

struct DronePose2D
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;
};

struct GridRoiConfig
{
    double cell_size_x = 0.5;
    double cell_size_y = 0.5;
    double margin_x = 0.10;
    double margin_y = 0.10;
};

struct ProjectionResult
{
    bool valid = false;
    Vec3 world_point;
    Vec3 ray_camera;
    Vec3 ray_body;
    Vec3 ray_world;
    double ray_scale = 0.0;
};

struct GridRoiResult
{
    bool inside = false;
    double dx = 0.0;
    double dy = 0.0;
    double limit_x = 0.0;
    double limit_y = 0.0;
};

struct GridProjectionFilterResult
{
    bool accepted = false;
    ProjectionResult projection;
    GridRoiResult roi;
};

inline bool isFinite(double value)
{
    return std::isfinite(value);
}

inline Vec3 add(const Vec3& a, const Vec3& b)
{
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 scale(const Vec3& v, double s)
{
    return Vec3{v.x * s, v.y * s, v.z * s};
}

inline Vec3 rotateYaw(const Vec3& v, double yaw)
{
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    return Vec3{
        c * v.x - s * v.y,
        s * v.x + c * v.y,
        v.z};
}

inline Vec3 multiply(const Matrix3& r, const Vec3& v)
{
    return Vec3{
        r.m[0] * v.x + r.m[1] * v.y + r.m[2] * v.z,
        r.m[3] * v.x + r.m[4] * v.y + r.m[5] * v.z,
        r.m[6] * v.x + r.m[7] * v.y + r.m[8] * v.z};
}

inline Matrix3 multiply(const Matrix3& a, const Matrix3& b)
{
    Matrix3 out;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            out.m[row * 3 + col] =
                a.m[row * 3 + 0] * b.m[0 * 3 + col] +
                a.m[row * 3 + 1] * b.m[1 * 3 + col] +
                a.m[row * 3 + 2] * b.m[2 * 3 + col];
        }
    }
    return out;
}

inline Matrix3 rotationX(double roll)
{
    const double c = std::cos(roll);
    const double s = std::sin(roll);
    Matrix3 r;
    r.m = {{1.0, 0.0, 0.0,
            0.0, c, -s,
            0.0, s, c}};
    return r;
}

inline Matrix3 rotationY(double pitch)
{
    const double c = std::cos(pitch);
    const double s = std::sin(pitch);
    Matrix3 r;
    r.m = {{c, 0.0, s,
            0.0, 1.0, 0.0,
            -s, 0.0, c}};
    return r;
}

inline Matrix3 rotationZ(double yaw)
{
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    Matrix3 r;
    r.m = {{c, -s, 0.0,
            s, c, 0.0,
            0.0, 0.0, 1.0}};
    return r;
}

inline Matrix3 cameraToBodyFromRpy(double roll, double pitch, double yaw)
{
    return multiply(rotationZ(yaw), multiply(rotationY(pitch), rotationX(roll)));
}

inline Matrix3 downwardCameraToBody(double body_x_from_image_down = -1.0,
                                    double body_y_from_image_right = -1.0)
{
    Matrix3 r;
    r.m = {{0.0, body_x_from_image_down, 0.0,
            body_y_from_image_right, 0.0, 0.0,
            0.0, 0.0, -1.0}};
    return r;
}

inline Vec3 undistortedCameraRay(const CameraIntrinsics& intrinsics, double pixel_u, double pixel_v)
{
    if (!isFinite(pixel_u) || !isFinite(pixel_v) ||
        std::abs(intrinsics.fx) < 1e-9 || std::abs(intrinsics.fy) < 1e-9)
    {
        return Vec3{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
    }

    const double xd = (pixel_u - intrinsics.cx) / intrinsics.fx;
    const double yd = (pixel_v - intrinsics.cy) / intrinsics.fy;
    double x = xd;
    double y = yd;

    if (intrinsics.use_distortion)
    {
        for (int iter = 0; iter < 8; ++iter)
        {
            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double radial =
                1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6;
            if (std::abs(radial) < 1e-9)
            {
                break;
            }

            const double delta_x =
                2.0 * intrinsics.p1 * x * y + intrinsics.p2 * (r2 + 2.0 * x * x);
            const double delta_y =
                intrinsics.p1 * (r2 + 2.0 * y * y) + 2.0 * intrinsics.p2 * x * y;
            x = (xd - delta_x) / radial;
            y = (yd - delta_y) / radial;
        }
    }

    return Vec3{x, y, 1.0};
}

inline ProjectionResult projectPixelToGround(
    const CameraProjectionConfig& config,
    const DronePose2D& drone_pose,
    double pixel_u,
    double pixel_v)
{
    ProjectionResult result;
    result.ray_camera = undistortedCameraRay(config.intrinsics, pixel_u, pixel_v);
    if (!isFinite(result.ray_camera.x) ||
        !isFinite(result.ray_camera.y) ||
        !isFinite(result.ray_camera.z))
    {
        return result;
    }

    result.ray_body = multiply(config.camera_to_body, result.ray_camera);
    result.ray_world = rotateYaw(result.ray_body, drone_pose.yaw);

    if (result.ray_world.z >= -config.min_ray_down_z)
    {
        return result;
    }

    const Vec3 camera_world = add(
        Vec3{drone_pose.x, drone_pose.y, drone_pose.z},
        rotateYaw(config.camera_offset_body, drone_pose.yaw));

    result.ray_scale = (config.ground_z_world - camera_world.z) / result.ray_world.z;
    if (result.ray_scale <= 0.0 || !isFinite(result.ray_scale))
    {
        return result;
    }

    result.world_point = add(camera_world, scale(result.ray_world, result.ray_scale));
    result.valid = true;
    return result;
}

inline ProjectionResult projectOffsetToGround(
    const CameraProjectionConfig& config,
    const DronePose2D& drone_pose,
    double pixel_dx,
    double pixel_dy)
{
    return projectPixelToGround(
        config,
        drone_pose,
        config.intrinsics.cx + pixel_dx,
        config.intrinsics.cy + pixel_dy);
}

inline GridRoiResult checkCurrentGridRoi(
    const GridRoiConfig& roi_config,
    const Vec3& world_point,
    double grid_center_x,
    double grid_center_y)
{
    GridRoiResult result;
    result.dx = world_point.x - grid_center_x;
    result.dy = world_point.y - grid_center_y;
    result.limit_x = roi_config.cell_size_x * 0.5 + roi_config.margin_x;
    result.limit_y = roi_config.cell_size_y * 0.5 + roi_config.margin_y;
    result.inside =
        std::abs(result.dx) <= result.limit_x &&
        std::abs(result.dy) <= result.limit_y;
    return result;
}

inline GridProjectionFilterResult filterCurrentGrid(
    const CameraProjectionConfig& projection_config,
    const GridRoiConfig& roi_config,
    const DronePose2D& drone_pose,
    double pixel_u,
    double pixel_v,
    double grid_center_x,
    double grid_center_y)
{
    GridProjectionFilterResult result;
    result.projection = projectPixelToGround(projection_config, drone_pose, pixel_u, pixel_v);
    if (!result.projection.valid)
    {
        return result;
    }

    result.roi = checkCurrentGridRoi(
        roi_config,
        result.projection.world_point,
        grid_center_x,
        grid_center_y);
    result.accepted = result.roi.inside;
    return result;
}

inline GridProjectionFilterResult filterCurrentGridByOffset(
    const CameraProjectionConfig& projection_config,
    const GridRoiConfig& roi_config,
    const DronePose2D& drone_pose,
    double pixel_dx,
    double pixel_dy,
    double grid_center_x,
    double grid_center_y)
{
    return filterCurrentGrid(
        projection_config,
        roi_config,
        drone_pose,
        projection_config.intrinsics.cx + pixel_dx,
        projection_config.intrinsics.cy + pixel_dy,
        grid_center_x,
        grid_center_y);
}

}  // namespace electronic_fly
