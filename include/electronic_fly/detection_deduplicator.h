// 功能：检测结果去重模块接口声明。

#pragma once

#include <map>
#include <string>
#include <vector>

#include <ros/time.h>
#include <ros/node_handle.h>

#include "electronic_fly/detection_types.h"

namespace electronic_fly
{

DedupConfig loadDedupConfig(ros::NodeHandle& pnh);

class DetectionDeduplicator
{
public:
    explicit DetectionDeduplicator(const DedupConfig& config);

    bool isDuplicateAtWithLimit(
        const std::string& label,
        double world_x,
        double world_y,
        double world_z,
        const std::string& current_grid_code,
        std::size_t history_limit) const;

    bool isDuplicateAt(
        const std::string& label,
        double world_x,
        double world_y,
        double world_z,
        const std::string& current_grid_code) const;

    void rememberAt(
        const std::string& label,
        double world_x,
        double world_y,
        double world_z,
        const std::string& grid_code);

    std::size_t historySize() const;

    const std::vector<AcceptedDetectionRecord>& history() const;
    const std::vector<AcceptedDetectionRecord>& missionRecords() const;

private:
    DedupConfig config_;
    std::vector<AcceptedDetectionRecord> accepted_detection_history_;
    std::vector<AcceptedDetectionRecord> mission_detection_records_;
};

std::string formatAcceptedDetectionsSummary(
    const std::vector<AcceptedDetectionRecord>& records);

std::string formatCompetitionReport(
    const std::string& status,
    const std::vector<std::string>& class_names,
    const std::map<std::string, int>& mission_totals,
    const std::vector<AcceptedDetectionRecord>& records);

}  // namespace electronic_fly
