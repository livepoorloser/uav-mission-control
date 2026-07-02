// 功能：检测结果去重，避免同一目标被重复计数。

#include "electronic_fly/detection_deduplicator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <ros/console.h>

#include "electronic_fly/detection_processor.h"

namespace electronic_fly
{

DedupConfig loadDedupConfig(ros::NodeHandle& pnh)
{
    DedupConfig config;
    config.enable = pnh.param("DetectionDedup/enable", true);
    config.max_position_delta_xy = pnh.param("DetectionDedup/max_position_delta_xy", 0.55);
    config.max_position_delta_z = pnh.param("DetectionDedup/max_position_delta_z", 0.35);
    config.cooldown_seconds = pnh.param("DetectionDedup/cooldown_seconds", 8.0);
    return config;
}

DetectionDeduplicator::DetectionDeduplicator(const DedupConfig& config)
    : config_(config)
{
}

bool DetectionDeduplicator::isDuplicateAtWithLimit(
    const std::string& label,
    double world_x,
    double world_y,
    double world_z,
    const std::string& current_grid_code,
    std::size_t history_limit) const
{
    if (!config_.enable)
    {
        return false;
    }

    const ros::Time now = ros::Time::now();
    const std::size_t checked_records = std::min(history_limit, accepted_detection_history_.size());
    for (std::size_t index = 0; index < checked_records; ++index)
    {
        const auto& record = accepted_detection_history_[index];
        if (record.label != label)
        {
            continue;
        }
        if ((now - record.stamp).toSec() > config_.cooldown_seconds)
        {
            continue;
        }

        const double dx = world_x - record.x;
        const double dy = world_y - record.y;
        const double dz = world_z - record.z;
        const double dist_xy = std::sqrt(dx * dx + dy * dy);
        if (dist_xy <= config_.max_position_delta_xy &&
            std::abs(dz) <= config_.max_position_delta_z)
        {
            ROS_INFO_THROTTLE(
                1.0,
                "mission_controller: duplicate detection suppressed label=%s current_grid=%s previous_grid=%s",
                label.c_str(),
                current_grid_code.c_str(),
                record.grid_code.c_str());
            return true;
        }
    }
    return false;
}

bool DetectionDeduplicator::isDuplicateAt(
    const std::string& label,
    double world_x,
    double world_y,
    double world_z,
    const std::string& current_grid_code) const
{
    return isDuplicateAtWithLimit(
        label,
        world_x,
        world_y,
        world_z,
        current_grid_code,
        accepted_detection_history_.size());
}

void DetectionDeduplicator::rememberAt(
    const std::string& label,
    double world_x,
    double world_y,
    double world_z,
    const std::string& grid_code)
{
    AcceptedDetectionRecord record;
    record.label = label;
    record.x = world_x;
    record.y = world_y;
    record.z = world_z;
    record.stamp = ros::Time::now();
    record.grid_code = grid_code;
    accepted_detection_history_.push_back(record);
    mission_detection_records_.push_back(record);

    const ros::Time now = ros::Time::now();
    accepted_detection_history_.erase(
        std::remove_if(
            accepted_detection_history_.begin(),
            accepted_detection_history_.end(),
            [&](const AcceptedDetectionRecord& item) {
                return (now - item.stamp).toSec() >
                    std::max(20.0, config_.cooldown_seconds * 3.0);
            }),
        accepted_detection_history_.end());
}

std::size_t DetectionDeduplicator::historySize() const
{
    return accepted_detection_history_.size();
}

const std::vector<AcceptedDetectionRecord>& DetectionDeduplicator::history() const
{
    return accepted_detection_history_;
}

const std::vector<AcceptedDetectionRecord>& DetectionDeduplicator::missionRecords() const
{
    return mission_detection_records_;
}

std::string formatAcceptedDetectionsSummary(
    const std::vector<AcceptedDetectionRecord>& records)
{
    if (records.empty())
    {
        return "none";
    }

    std::ostringstream stream;
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        const auto& record = records[index];
        if (index > 0)
        {
            stream << " | ";
        }
        stream << record.label << " 1 (("
               << std::fixed << std::setprecision(2)
               << record.x << "," << record.y << "," << record.z << ") ";
        if (!record.grid_code.empty())
        {
            stream << record.grid_code;
        }
        else
        {
            stream << "unknown_grid";
        }
        stream << ")";
    }
    return stream.str();
}

std::string formatCompetitionReport(
    const std::string& status,
    const std::vector<std::string>& class_names,
    const std::map<std::string, int>& mission_totals,
    const std::vector<AcceptedDetectionRecord>& records)
{
    std::ostringstream report;
    report << "\n========== Competition Inspection Report ==========\n";
    report << "Status: " << status << "\n";
    report << "Animal Total: " << sumCounts(mission_totals) << "\n";
    report << "Class Counts:\n";
    for (const auto& name : class_names)
    {
        const auto found = mission_totals.find(name);
        report << "  " << name << ": " << (found == mission_totals.end() ? 0 : found->second) << "\n";
    }

    report << "Detections:\n";
    if (records.empty())
    {
        report << "  none\n";
    }
    else
    {
        std::map<std::string, int> label_indices;
        for (const auto& record : records)
        {
            const int index = ++label_indices[record.label];
            report << "  " << record.label << index << ": ";
            if (!record.grid_code.empty())
            {
                report << record.grid_code;
            }
            else
            {
                report << "unknown_grid";
            }
            report << " ("
                   << std::fixed << std::setprecision(2)
                   << record.x << ", " << record.y << ", " << record.z << ")\n";
        }
    }
    report << "===================================================";
    return report.str();
}

}  // namespace electronic_fly
