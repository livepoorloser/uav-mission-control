// 功能：视觉检测、投影和任务统计的数据结构定义。

#pragma once

#include <map>
#include <string>
#include <vector>

#include <ros/time.h>

#include "electronic_fly/grid_projection_filter.h"

namespace electronic_fly
{

struct DetectionSnapshot
{
    std::map<std::string, int> counts;
    int total = 0;
    double timestamp = 0.0;
    bool tracking_valid = false;
    double avg_dx = 0.0;
    double avg_dy = 0.0;
    double primary_dx = 0.0;
    double primary_dy = 0.0;
    double primary_score = 0.0;
    double primary_area_px = 0.0;
    bool primary_has_pixel = false;
    double primary_u = 0.0;
    double primary_v = 0.0;
    double primary_raw_u = 0.0;
    double primary_raw_v = 0.0;
    bool primary_projection_valid = false;
    double primary_world_x = 0.0;
    double primary_world_y = 0.0;
    double primary_world_z = 0.0;
    std::string primary_grid_code;
    bool primary_current_grid_match = false;
    std::string primary_label;

    struct DetectionItem
    {
        std::string label;
        double dx = 0.0;
        double dy = 0.0;
        bool has_pixel = false;
        double u = 0.0;
        double v = 0.0;
        double raw_u = 0.0;
        double raw_v = 0.0;
        bool undistorted = false;
        double score = 0.0;
        double area_px = 0.0;
        bool projection_valid = false;
        double world_x = 0.0;
        double world_y = 0.0;
        double world_z = 0.0;
        std::string projected_grid_code;
        bool current_grid_match = false;
    };

    std::vector<DetectionItem> detections;
};

struct DetectionFilterConfig
{
    bool enable = true;
    bool use_primary_target = true;
    bool cluster_same_class = true;
    double max_abs_dx = 120.0;
    double max_abs_dy = 90.0;
    double max_radius_px = 150.0;
    double min_score = 0.0;
    double min_area_px = 0.0;
    double cluster_radius_px = 45.0;
};

struct CameraProjectionRuntimeConfig
{
    bool projection_enable = false;
    bool filter_enable = false;
    bool reject_outside_current_grid = true;
    bool reject_invalid_projection = false;
    CameraProjectionConfig projection;
    GridRoiConfig roi;
};

struct DedupConfig
{
    bool enable = true;
    double max_position_delta_xy = 0.55;
    double max_position_delta_z = 0.35;
    double cooldown_seconds = 8.0;
};

struct AcceptedDetectionRecord
{
    std::string label;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string grid_code;
    ros::Time stamp;
};

}  // namespace electronic_fly
