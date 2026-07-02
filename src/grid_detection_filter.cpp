// 功能：按网格和像素约束过滤检测结果。

#include "electronic_fly/grid_detection_filter.h"

#include <cmath>

#include <ros/console.h>

#include "electronic_fly/detection_processor.h"

namespace electronic_fly
{

namespace
{

double degToRad(double degrees)
{
    return degrees * std::acos(-1.0) / 180.0;
}

}  // namespace

DetectionFilterConfig loadDetectionFilterConfig(ros::NodeHandle& pnh)
{
    DetectionFilterConfig config;
    config.enable = pnh.param("DetectionFilter/enable", true);
    config.use_primary_target = pnh.param("DetectionFilter/use_primary_target", true);
    config.cluster_same_class = pnh.param("DetectionFilter/cluster_same_class", true);
    config.max_abs_dx = pnh.param("DetectionFilter/max_abs_dx", 120.0);
    config.max_abs_dy = pnh.param("DetectionFilter/max_abs_dy", 90.0);
    config.max_radius_px = pnh.param("DetectionFilter/max_radius_px", 150.0);
    config.min_score = pnh.param("DetectionFilter/min_score", 0.0);
    config.min_area_px = pnh.param("DetectionFilter/min_area_px", 0.0);
    config.cluster_radius_px = pnh.param("DetectionFilter/cluster_radius_px", 45.0);
    return config;
}

CameraProjectionRuntimeConfig loadCameraProjectionConfig(ros::NodeHandle& pnh)
{
    CameraProjectionRuntimeConfig config;
    config.projection_enable = pnh.param("CameraProjection/enable", false);
    config.projection.intrinsics.image_width = pnh.param("CameraProjection/image_width", 640);
    config.projection.intrinsics.image_height = pnh.param("CameraProjection/image_height", 480);
    config.projection.intrinsics.fx = pnh.param("CameraProjection/fx", 457.0);
    config.projection.intrinsics.fy = pnh.param("CameraProjection/fy", 457.0);
    config.projection.intrinsics.cx = pnh.param("CameraProjection/cx", 320.0);
    config.projection.intrinsics.cy = pnh.param("CameraProjection/cy", 240.0);
    config.projection.intrinsics.k1 = pnh.param("CameraProjection/k1", 0.0);
    config.projection.intrinsics.k2 = pnh.param("CameraProjection/k2", 0.0);
    config.projection.intrinsics.p1 = pnh.param("CameraProjection/p1", 0.0);
    config.projection.intrinsics.p2 = pnh.param("CameraProjection/p2", 0.0);
    config.projection.intrinsics.k3 = pnh.param("CameraProjection/k3", 0.0);
    config.projection.intrinsics.use_distortion = pnh.param("CameraProjection/use_distortion", false);
    config.projection.ground_z_world = pnh.param("CameraProjection/ground_z_world", 0.0);
    config.projection.min_ray_down_z = pnh.param("CameraProjection/min_ray_down_z", 1e-3);
    config.projection.camera_offset_body.x = pnh.param("CameraProjection/camera_offset_x", 0.025);
    config.projection.camera_offset_body.y = pnh.param("CameraProjection/camera_offset_y", 0.0);
    config.projection.camera_offset_body.z = pnh.param("CameraProjection/camera_offset_z", 0.0);

    const bool use_downward_mount = pnh.param("CameraProjection/use_downward_mount", true);
    if (use_downward_mount)
    {
        const double body_x_from_image_down =
            pnh.param("CameraProjection/body_x_from_image_down", -1.0);
        const double body_y_from_image_right =
            pnh.param("CameraProjection/body_y_from_image_right", -1.0);
        config.projection.camera_to_body =
            downwardCameraToBody(body_x_from_image_down, body_y_from_image_right);
    }
    else
    {
        config.projection.camera_to_body = cameraToBodyFromRpy(
            degToRad(pnh.param("CameraProjection/camera_roll_deg", 0.0)),
            degToRad(pnh.param("CameraProjection/camera_pitch_deg", 0.0)),
            degToRad(pnh.param("CameraProjection/camera_yaw_deg", 0.0)));
    }

    const double correction_roll = degToRad(pnh.param("CameraProjection/correction_roll_deg", 0.0));
    const double correction_pitch = degToRad(pnh.param("CameraProjection/correction_pitch_deg", 0.0));
    const double correction_yaw = degToRad(pnh.param("CameraProjection/correction_yaw_deg", 0.0));
    const bool has_correction =
        std::abs(correction_roll) > 1e-9 ||
        std::abs(correction_pitch) > 1e-9 ||
        std::abs(correction_yaw) > 1e-9;
    if (has_correction)
    {
        config.projection.camera_to_body = multiply(
            cameraToBodyFromRpy(correction_roll, correction_pitch, correction_yaw),
            config.projection.camera_to_body);
    }

    config.filter_enable = pnh.param("GridProjectionFilter/enable", false);
    config.reject_outside_current_grid =
        pnh.param("GridProjectionFilter/reject_outside_current_grid", true);
    config.reject_invalid_projection =
        pnh.param("GridProjectionFilter/reject_invalid_projection", false);
    config.roi.cell_size_x = pnh.param("GridProjectionFilter/cell_size_x", 0.5);
    config.roi.cell_size_y = pnh.param("GridProjectionFilter/cell_size_y", 0.5);
    config.roi.margin_x = pnh.param("GridProjectionFilter/margin_x", 0.10);
    config.roi.margin_y = pnh.param("GridProjectionFilter/margin_y", 0.10);

    if (config.projection_enable && config.filter_enable)
    {
        ROS_INFO(
            "grid_projection_filter: enabled fx=%.2f fy=%.2f cx=%.2f cy=%.2f distortion=%s offset=(%.3f, %.3f, %.3f) roi=(%.2f x %.2f + %.2f/%.2f)",
            config.projection.intrinsics.fx,
            config.projection.intrinsics.fy,
            config.projection.intrinsics.cx,
            config.projection.intrinsics.cy,
            config.projection.intrinsics.use_distortion ? "on" : "off",
            config.projection.camera_offset_body.x,
            config.projection.camera_offset_body.y,
            config.projection.camera_offset_body.z,
            config.roi.cell_size_x,
            config.roi.cell_size_y,
            config.roi.margin_x,
            config.roi.margin_y);
    }
    return config;
}

DronePose2D makeDronePose2D(
    const geometry_msgs::PoseStamped& pose,
    const std::function<double(const geometry_msgs::Quaternion&)>& yaw_from_quaternion)
{
    DronePose2D drone_pose;
    drone_pose.x = pose.pose.position.x;
    drone_pose.y = pose.pose.position.y;
    drone_pose.z = pose.pose.position.z;
    drone_pose.yaw = yaw_from_quaternion(pose.pose.orientation);
    return drone_pose;
}

DetectionSnapshot applyProjectionFilter(
    const DetectionSnapshot& snapshot,
    const Waypoint& current_waypoint,
    const geometry_msgs::PoseStamped& pose,
    const CameraProjectionRuntimeConfig& config,
    const std::vector<std::string>& class_names,
    const std::vector<double>& grid_center,
    const std::function<bool(const Vec3&, std::string&)>& grid_code_for_world_point,
    const std::function<double(const geometry_msgs::Quaternion&)>& yaw_from_quaternion)
{
    if (!config.projection_enable ||
        snapshot.total <= 0 ||
        snapshot.detections.empty() ||
        current_waypoint.pos.size() != 3 ||
        grid_center.size() != 3)
    {
        return snapshot;
    }

    DetectionSnapshot projected = snapshot;
    projected.detections.clear();
    const auto drone_pose = makeDronePose2D(pose, yaw_from_quaternion);
    for (auto detection : snapshot.detections)
    {
        double pixel_u =
            config.projection.intrinsics.image_width * 0.5 + detection.dx;
        double pixel_v =
            config.projection.intrinsics.image_height * 0.5 + detection.dy;
        if (detection.has_pixel)
        {
            if (config.projection.intrinsics.use_distortion)
            {
                pixel_u = detection.raw_u;
                pixel_v = detection.raw_v;
            }
            else
            {
                pixel_u = detection.u;
                pixel_v = detection.v;
            }
        }

        const auto filter_result = filterCurrentGrid(
            config.projection,
            config.roi,
            drone_pose,
            pixel_u,
            pixel_v,
            grid_center[0],
            grid_center[1]);

        detection.projection_valid = filter_result.projection.valid;
        if (filter_result.projection.valid)
        {
            detection.world_x = filter_result.projection.world_point.x;
            detection.world_y = filter_result.projection.world_point.y;
            detection.world_z = filter_result.projection.world_point.z;
            std::string projected_grid_code;
            if (grid_code_for_world_point(filter_result.projection.world_point, projected_grid_code))
            {
                detection.projected_grid_code = projected_grid_code;
            }
            if (!current_waypoint.grid_code.empty() && !detection.projected_grid_code.empty())
            {
                detection.current_grid_match =
                    detection.projected_grid_code == current_waypoint.grid_code;
            }
            else
            {
                detection.current_grid_match = filter_result.roi.inside;
            }
        }

        bool keep_detection = true;
        if (config.filter_enable)
        {
            if (filter_result.projection.valid)
            {
                keep_detection =
                    !config.reject_outside_current_grid ||
                    detection.current_grid_match;
            }
            else
            {
                keep_detection = !config.reject_invalid_projection;
            }
        }

        if (keep_detection)
        {
            projected.detections.push_back(detection);
        }
    }

    recomputeSnapshotFromDetections(projected, class_names);
    return projected;
}

bool isDetectionAccepted(
    const DetectionSnapshot& snapshot,
    const DetectionFilterConfig& config)
{
    if (!config.enable)
    {
        return snapshot.total > 0;
    }

    if (!snapshot.tracking_valid || snapshot.total <= 0)
    {
        return false;
    }

    const double dx =
        config.use_primary_target ? snapshot.primary_dx : snapshot.avg_dx;
    const double dy =
        config.use_primary_target ? snapshot.primary_dy : snapshot.avg_dy;
    const double radius = std::sqrt(dx * dx + dy * dy);

    if (std::abs(dx) > config.max_abs_dx)
    {
        return false;
    }
    if (std::abs(dy) > config.max_abs_dy)
    {
        return false;
    }
    if (radius > config.max_radius_px)
    {
        return false;
    }
    if (snapshot.primary_score < config.min_score)
    {
        return false;
    }
    if (snapshot.primary_area_px < config.min_area_px)
    {
        return false;
    }
    return true;
}

}  // namespace electronic_fly
