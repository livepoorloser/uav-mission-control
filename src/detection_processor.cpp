// 功能：解析和整理视觉检测结果，生成任务可用的检测快照。

#include "electronic_fly/detection_processor.h"

#include <cmath>
#include <sstream>

#include "electronic_fly/json_utils.h"

namespace electronic_fly
{

std::map<std::string, int> makeZeroCounts(const std::vector<std::string>& class_names)
{
    std::map<std::string, int> counts;
    for (const auto& name : class_names)
    {
        counts[name] = 0;
    }
    return counts;
}

int sumCounts(const std::map<std::string, int>& counts)
{
    int total = 0;
    for (const auto& item : counts)
    {
        total += item.second;
    }
    return total;
}

std::string formatCountsSummary(
    const std::vector<std::string>& class_names,
    const std::map<std::string, int>& counts,
    bool include_zero)
{
    std::ostringstream stream;
    bool first = true;
    for (const auto& name : class_names)
    {
        const auto found = counts.find(name);
        const int value = found == counts.end() ? 0 : found->second;
        if (!include_zero && value <= 0)
        {
            continue;
        }

        if (!first)
        {
            stream << " ";
        }
        stream << name << ":" << value;
        first = false;
    }

    if (first)
    {
        return "none";
    }
    return stream.str();
}

boost::property_tree::ptree loadDetectionSummaryFallback(ros::NodeHandle& nh)
{
    boost::property_tree::ptree tree;
    std::string raw_json;
    if (nh.getParam("vision/last_detection_json", raw_json))
    {
        parseJson(raw_json, tree);
    }
    return tree;
}

DetectionSnapshot getDetectionSnapshot(
    ros::NodeHandle& nh,
    const boost::property_tree::ptree& latest_detection_summary,
    bool have_detection_summary,
    const std::vector<std::string>& class_names,
    double detection_start_sec)
{
    DetectionSnapshot snapshot;
    snapshot.counts = makeZeroCounts(class_names);

    boost::property_tree::ptree summary;
    if (have_detection_summary)
    {
        summary = latest_detection_summary;
    }
    else
    {
        summary = loadDetectionSummaryFallback(nh);
    }

    if (summary.empty())
    {
        return snapshot;
    }

    snapshot.timestamp = summary.get<double>("timestamp", 0.0);
    if (snapshot.timestamp + 1e-6 < detection_start_sec)
    {
        return snapshot;
    }

    for (const auto& name : class_names)
    {
        snapshot.counts[name] = summary.get<int>("counts." + name, 0);
    }

    snapshot.total = summary.get<int>("total", 0);
    if (snapshot.total <= 0)
    {
        snapshot.total = sumCounts(snapshot.counts);
    }

    snapshot.tracking_valid = summary.get<bool>("tracking.valid", false);
    snapshot.avg_dx = summary.get<double>("tracking.avg_dx", 0.0);
    snapshot.avg_dy = summary.get<double>("tracking.avg_dy", 0.0);
    snapshot.primary_dx = summary.get<double>("tracking.primary_dx", 0.0);
    snapshot.primary_dy = summary.get<double>("tracking.primary_dy", 0.0);
    snapshot.primary_score = summary.get<double>("tracking.primary_score", 0.0);
    snapshot.primary_area_px = summary.get<double>("tracking.primary_area_px", 0.0);
    snapshot.primary_label = summary.get<std::string>("tracking.primary_label", "");
    snapshot.primary_u = summary.get<double>("tracking.primary_u", 0.0);
    snapshot.primary_v = summary.get<double>("tracking.primary_v", 0.0);
    snapshot.primary_raw_u = summary.get<double>("tracking.primary_raw_u", snapshot.primary_u);
    snapshot.primary_raw_v = summary.get<double>("tracking.primary_raw_v", snapshot.primary_v);
    const auto tracking_primary_u = summary.get_optional<double>("tracking.primary_u");
    const auto tracking_primary_v = summary.get_optional<double>("tracking.primary_v");
    snapshot.primary_has_pixel = static_cast<bool>(tracking_primary_u) && static_cast<bool>(tracking_primary_v);

    const auto diffs_child = summary.get_child_optional("center_diffs");
    if (diffs_child)
    {
        for (const auto& item : *diffs_child)
        {
            DetectionSnapshot::DetectionItem detection;
            detection.label = item.second.get<std::string>("class", "");
            detection.dx = item.second.get<double>("dx", 0.0);
            detection.dy = item.second.get<double>("dy", 0.0);
            const auto pixel_u = item.second.get_optional<double>("u");
            const auto pixel_v = item.second.get_optional<double>("v");
            detection.has_pixel = static_cast<bool>(pixel_u) && static_cast<bool>(pixel_v);
            detection.u = item.second.get<double>("u", 0.0);
            detection.v = item.second.get<double>("v", 0.0);
            detection.raw_u = item.second.get<double>("raw_u", detection.u);
            detection.raw_v = item.second.get<double>("raw_v", detection.v);
            detection.undistorted = item.second.get<bool>("undistorted", false);
            detection.score = item.second.get<double>("score", 0.0);
            detection.area_px = item.second.get<double>("area_px", 0.0);
            if (!detection.label.empty())
            {
                snapshot.detections.push_back(detection);
            }
        }
    }
    return snapshot;
}

void recomputeSnapshotFromDetections(
    DetectionSnapshot& snapshot,
    const std::vector<std::string>& class_names)
{
    snapshot.counts = makeZeroCounts(class_names);
    snapshot.total = 0;
    snapshot.tracking_valid = false;
    snapshot.avg_dx = 0.0;
    snapshot.avg_dy = 0.0;
    snapshot.primary_dx = 0.0;
    snapshot.primary_dy = 0.0;
    snapshot.primary_score = 0.0;
    snapshot.primary_area_px = 0.0;
    snapshot.primary_has_pixel = false;
    snapshot.primary_u = 0.0;
    snapshot.primary_v = 0.0;
    snapshot.primary_raw_u = 0.0;
    snapshot.primary_raw_v = 0.0;
    snapshot.primary_projection_valid = false;
    snapshot.primary_world_x = 0.0;
    snapshot.primary_world_y = 0.0;
    snapshot.primary_world_z = 0.0;
    snapshot.primary_grid_code.clear();
    snapshot.primary_current_grid_match = false;
    snapshot.primary_label.clear();

    if (snapshot.detections.empty())
    {
        return;
    }

    double sum_dx = 0.0;
    double sum_dy = 0.0;
    bool have_primary = false;
    DetectionSnapshot::DetectionItem primary;
    for (const auto& detection : snapshot.detections)
    {
        if (detection.label.empty())
        {
            continue;
        }
        snapshot.counts[detection.label] += 1;
        snapshot.total += 1;
        sum_dx += detection.dx;
        sum_dy += detection.dy;
        if (!have_primary ||
            detection.score > primary.score ||
            (std::abs(detection.score - primary.score) < 1e-6 &&
             detection.area_px > primary.area_px))
        {
            primary = detection;
            have_primary = true;
        }
    }

    if (snapshot.total <= 0 || !have_primary)
    {
        return;
    }

    snapshot.avg_dx = sum_dx / static_cast<double>(snapshot.total);
    snapshot.avg_dy = sum_dy / static_cast<double>(snapshot.total);
    snapshot.tracking_valid = true;
    snapshot.primary_label = primary.label;
    snapshot.primary_dx = primary.dx;
    snapshot.primary_dy = primary.dy;
    snapshot.primary_score = primary.score;
    snapshot.primary_area_px = primary.area_px;
    snapshot.primary_has_pixel = primary.has_pixel;
    snapshot.primary_u = primary.u;
    snapshot.primary_v = primary.v;
    snapshot.primary_raw_u = primary.raw_u;
    snapshot.primary_raw_v = primary.raw_v;
    snapshot.primary_projection_valid = primary.projection_valid;
    snapshot.primary_world_x = primary.world_x;
    snapshot.primary_world_y = primary.world_y;
    snapshot.primary_world_z = primary.world_z;
    snapshot.primary_grid_code = primary.projected_grid_code;
    snapshot.primary_current_grid_match = primary.current_grid_match;
}

std::string snapshotProjectedDetectionsToJson(
    const DetectionSnapshot& snapshot,
    const std::string& current_grid_code)
{
    boost::property_tree::ptree root;
    root.put("timestamp", snapshot.timestamp);
    root.put("current_grid", current_grid_code);
    root.put("total", snapshot.total);
    boost::property_tree::ptree detections_tree;
    for (const auto& detection : snapshot.detections)
    {
        boost::property_tree::ptree item;
        item.put("class", detection.label);
        item.put("score", detection.score);
        item.put("area_px", detection.area_px);
        item.put("dx", detection.dx);
        item.put("dy", detection.dy);
        item.put("u", detection.u);
        item.put("v", detection.v);
        item.put("projection_valid", detection.projection_valid);
        item.put("world_x", detection.world_x);
        item.put("world_y", detection.world_y);
        item.put("world_z", detection.world_z);
        item.put("projected_grid", detection.projected_grid_code);
        item.put("current_grid_match", detection.current_grid_match);
        appendArrayItem(detections_tree, item);
    }
    root.add_child("detections", detections_tree);
    return toJson(root, false);
}

DetectionSnapshot clusterSnapshotDetections(
    const DetectionSnapshot& snapshot,
    const std::vector<std::string>& class_names,
    const DetectionFilterConfig& config)
{
    if (!config.cluster_same_class || snapshot.detections.empty() || config.cluster_radius_px <= 0.0)
    {
        return snapshot;
    }

    DetectionSnapshot clustered = snapshot;
    clustered.detections.clear();

    std::map<std::string, std::vector<DetectionSnapshot::DetectionItem>> clusters;
    const double radius_sq = config.cluster_radius_px * config.cluster_radius_px;

    for (const auto& detection : snapshot.detections)
    {
        if (detection.score < config.min_score || detection.area_px < config.min_area_px)
        {
            continue;
        }

        auto& label_clusters = clusters[detection.label];
        bool merged = false;
        for (auto& cluster : label_clusters)
        {
            const double dx = detection.dx - cluster.dx;
            const double dy = detection.dy - cluster.dy;
            if (dx * dx + dy * dy <= radius_sq)
            {
                if (detection.score > cluster.score ||
                    (std::abs(detection.score - cluster.score) < 1e-6 &&
                     detection.area_px > cluster.area_px))
                {
                    cluster = detection;
                }
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            label_clusters.push_back(detection);
        }
    }

    for (const auto& item : clusters)
    {
        for (const auto& detection : item.second)
        {
            clustered.detections.push_back(detection);
        }
    }
    recomputeSnapshotFromDetections(clustered, class_names);
    return clustered;
}

}  // namespace electronic_fly
