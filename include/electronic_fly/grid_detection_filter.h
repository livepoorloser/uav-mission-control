// 功能：网格检测过滤模块接口声明。

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <ros/node_handle.h>

#include "electronic_fly/detection_types.h"
#include "electronic_fly/trajectory_optimizer/trajectory_optimizer.h"

namespace electronic_fly
{

DetectionFilterConfig loadDetectionFilterConfig(ros::NodeHandle& pnh);

CameraProjectionRuntimeConfig loadCameraProjectionConfig(ros::NodeHandle& pnh);

DronePose2D makeDronePose2D(
    const geometry_msgs::PoseStamped& pose,
    const std::function<double(const geometry_msgs::Quaternion&)>& yaw_from_quaternion);

DetectionSnapshot applyProjectionFilter(
    const DetectionSnapshot& snapshot,
    const Waypoint& current_waypoint,
    const geometry_msgs::PoseStamped& pose,
    const CameraProjectionRuntimeConfig& config,
    const std::vector<std::string>& class_names,
    const std::vector<double>& grid_center,
    const std::function<bool(const Vec3&, std::string&)>& grid_code_for_world_point,
    const std::function<double(const geometry_msgs::Quaternion&)>& yaw_from_quaternion);

bool isDetectionAccepted(
    const DetectionSnapshot& snapshot,
    const DetectionFilterConfig& config);

}  // namespace electronic_fly
