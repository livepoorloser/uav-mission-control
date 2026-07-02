// 功能：视觉检测结果解析和快照生成接口声明。

#pragma once

#include <map>
#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <ros/node_handle.h>

#include "electronic_fly/detection_types.h"

namespace electronic_fly
{

std::map<std::string, int> makeZeroCounts(const std::vector<std::string>& class_names);

int sumCounts(const std::map<std::string, int>& counts);

std::string formatCountsSummary(
    const std::vector<std::string>& class_names,
    const std::map<std::string, int>& counts,
    bool include_zero = false);

boost::property_tree::ptree loadDetectionSummaryFallback(ros::NodeHandle& nh);

DetectionSnapshot getDetectionSnapshot(
    ros::NodeHandle& nh,
    const boost::property_tree::ptree& latest_detection_summary,
    bool have_detection_summary,
    const std::vector<std::string>& class_names,
    double detection_start_sec);

void recomputeSnapshotFromDetections(
    DetectionSnapshot& snapshot,
    const std::vector<std::string>& class_names);

std::string snapshotProjectedDetectionsToJson(
    const DetectionSnapshot& snapshot,
    const std::string& current_grid_code);

DetectionSnapshot clusterSnapshotDetections(
    const DetectionSnapshot& snapshot,
    const std::vector<std::string>& class_names,
    const DetectionFilterConfig& config);

}  // namespace electronic_fly
