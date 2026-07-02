// 功能：根据赛场网格和障碍信息生成巡检路径与返航路径。

#include <ros/ros.h>
#include <XmlRpcValue.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "electronic_fly/grid_utils.h"

namespace
{

using electronic_fly::GridCell;

struct PlannedPath
{
    std::vector<GridCell> survey_cells;
    std::vector<int> inspect_flags;
    std::vector<GridCell> return_cells;
};

void setEmptyOutputs(ros::NodeHandle& nh)
{
    nh.setParam("/path_planning_ready", false);
    nh.setParam("/path_planning_x", std::vector<int>());
    nh.setParam("/path_planning_y", std::vector<int>());
    nh.setParam("/path_planning_rows", std::vector<int>());
    nh.setParam("/path_planning_cols", std::vector<int>());
    nh.setParam("/path_planning_inspect_flags", std::vector<int>());
    nh.setParam("/return_path_planning_x", std::vector<int>());
    nh.setParam("/return_path_planning_y", std::vector<int>());
    nh.setParam("/return_path_rows", std::vector<int>());
    nh.setParam("/return_path_cols", std::vector<int>());
    nh.setParam("vision/path_planning_x", std::vector<int>());
    nh.setParam("vision/path_planning_y", std::vector<int>());
    nh.setParam("vision/path_planning_inspect_flags", std::vector<int>());
    nh.setParam("vision/return_path_planning_x", std::vector<int>());
    nh.setParam("vision/return_path_planning_y", std::vector<int>());
}

bool parseCellFromValue(const XmlRpc::XmlRpcValue& value, int rows, int cols, GridCell& cell)
{
    try
    {
        if (value.getType() == XmlRpc::XmlRpcValue::TypeString)
        {
            return electronic_fly::parseGridCode(static_cast<std::string>(value), cell) &&
                   electronic_fly::isInsideGrid(cell, rows, cols);
        }

        if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
            const int flattened = static_cast<int>(value);
            if (flattened < 0 || flattened >= rows * cols)
            {
                return false;
            }
            cell.row = flattened / cols;
            cell.col = flattened % cols;
            return true;
        }

        if (value.getType() == XmlRpc::XmlRpcValue::TypeArray && value.size() >= 2)
        {
            const auto read_number = [](const XmlRpc::XmlRpcValue& number) -> int {
                if (number.getType() == XmlRpc::XmlRpcValue::TypeInt)
                {
                    return static_cast<int>(number);
                }
                if (number.getType() == XmlRpc::XmlRpcValue::TypeDouble)
                {
                    return static_cast<int>(static_cast<double>(number));
                }
                throw std::runtime_error("non numeric grid cell");
            };

            const int first = read_number(value[0]);
            const int second = read_number(value[1]);

            GridCell candidate;
            candidate.row = first;
            candidate.col = second;
            if (electronic_fly::isInsideGrid(candidate, rows, cols))
            {
                cell = candidate;
                return true;
            }

            candidate.row = second;
            candidate.col = first;
            if (electronic_fly::isInsideGrid(candidate, rows, cols))
            {
                cell = candidate;
                return true;
            }
        }
    }
    catch (...)
    {
        return false;
    }

    return false;
}

std::vector<GridCell> loadBlockedCells(ros::NodeHandle& nh, ros::NodeHandle& pnh, int rows, int cols)
{
    const std::string source = pnh.param<std::string>("shelter_source", "param");
    const double wait_timeout = pnh.param("shelter_wait_timeout", 2.0);

    XmlRpc::XmlRpcValue raw_cells;
    bool have_cells = false;

    if (source == "param")
    {
        const ros::Time deadline = ros::Time::now() + ros::Duration(wait_timeout);
        while (ros::ok() && ros::Time::now() < deadline)
        {
            if (nh.getParam("shelter_location", raw_cells) &&
                raw_cells.getType() == XmlRpc::XmlRpcValue::TypeArray &&
                raw_cells.size() > 0)
            {
                have_cells = true;
                break;
            }
            ros::Duration(0.1).sleep();
        }
    }

    if (!have_cells)
    {
        if (pnh.getParam("manual_shelter_locations", raw_cells) &&
            raw_cells.getType() == XmlRpc::XmlRpcValue::TypeArray)
        {
            have_cells = true;
        }
    }

    std::vector<GridCell> blocked;
    if (!have_cells)
    {
        ROS_WARN("path_planning: no shelter list received, planning full sweep");
        return blocked;
    }

    std::set<std::pair<int, int>> unique_cells;
    for (int index = 0; index < raw_cells.size(); ++index)
    {
        GridCell cell;
        if (parseCellFromValue(raw_cells[index], rows, cols, cell))
        {
            unique_cells.insert(std::make_pair(cell.row, cell.col));
        }
        else
        {
            ROS_WARN_STREAM("path_planning: ignored invalid shelter entry at index " << index);
        }
    }

    for (const auto& item : unique_cells)
    {
        blocked.push_back(GridCell{item.first, item.second});
    }
    return blocked;
}

std::vector<GridCell> loadLandingCandidates(ros::NodeHandle& pnh, int rows, int cols)
{
    const bool return_to_takeoff = pnh.param("return_to_takeoff", true);
    if (return_to_takeoff)
    {
        XmlRpc::XmlRpcValue raw_takeoff_cell;
        GridCell takeoff_cell{0, 0};
        if (pnh.getParam("takeoff_cell", raw_takeoff_cell))
        {
            GridCell parsed_cell;
            if (parseCellFromValue(raw_takeoff_cell, rows, cols, parsed_cell))
            {
                takeoff_cell = parsed_cell;
            }
            else
            {
                ROS_WARN("path_planning: invalid takeoff_cell, falling back to [0, 0]");
            }
        }
        return {takeoff_cell};
    }

    XmlRpc::XmlRpcValue raw_candidates;
    std::vector<GridCell> candidates;
    if (!pnh.getParam("landing_candidates", raw_candidates) ||
        raw_candidates.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
        candidates.push_back(GridCell{0, 2});
        candidates.push_back(GridCell{2, 0});
        return candidates;
    }

    for (int index = 0; index < raw_candidates.size(); ++index)
    {
        GridCell cell;
        if (parseCellFromValue(raw_candidates[index], rows, cols, cell))
        {
            candidates.push_back(cell);
        }
    }

    if (candidates.empty())
    {
        candidates.push_back(GridCell{0, 2});
        candidates.push_back(GridCell{2, 0});
    }
    return candidates;
}

std::vector<GridCell> buildSurveyTargets(int rows, int cols, const std::vector<std::vector<bool>>& blocked)
{
    std::vector<GridCell> targets;
    for (int row = 0; row < rows; ++row)
    {
        if (row % 2 == 0)
        {
            for (int col = 0; col < cols; ++col)
            {
                if (!blocked[row][col])
                {
                    targets.push_back(GridCell{row, col});
                }
            }
        }
        else
        {
            for (int col = cols - 1; col >= 0; --col)
            {
                if (!blocked[row][col])
                {
                    targets.push_back(GridCell{row, col});
                }
            }
        }
    }
    return targets;
}

std::vector<GridCell> bfsPath(
    const GridCell& start,
    const GridCell& goal,
    int rows,
    int cols,
    const std::vector<std::vector<bool>>& blocked)
{
    if (start == goal)
    {
        return {start};
    }

    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<GridCell>> parent(rows, std::vector<GridCell>(cols, GridCell{-1, -1}));
    std::deque<GridCell> queue;
    queue.push_back(start);
    visited[start.row][start.col] = true;

    const int d_row[4] = {-1, 1, 0, 0};
    const int d_col[4] = {0, 0, -1, 1};

    while (!queue.empty())
    {
        const GridCell current = queue.front();
        queue.pop_front();

        for (int direction = 0; direction < 4; ++direction)
        {
            GridCell next{current.row + d_row[direction], current.col + d_col[direction]};
            if (!electronic_fly::isInsideGrid(next, rows, cols) ||
                blocked[next.row][next.col] ||
                visited[next.row][next.col])
            {
                continue;
            }

            visited[next.row][next.col] = true;
            parent[next.row][next.col] = current;
            if (next == goal)
            {
                std::vector<GridCell> path;
                GridCell trace = next;
                while (trace != start)
                {
                    path.push_back(trace);
                    trace = parent[trace.row][trace.col];
                }
                path.push_back(start);
                std::reverse(path.begin(), path.end());
                return path;
            }
            queue.push_back(next);
        }
    }

    return {};
}

void appendPathSegment(
    const std::vector<GridCell>& segment,
    bool inspect_last,
    std::vector<GridCell>& output_cells,
    std::vector<int>& inspect_flags)
{
    for (std::size_t index = 0; index < segment.size(); ++index)
    {
        const GridCell& cell = segment[index];
        const bool is_last = (index + 1 == segment.size());

        if (!output_cells.empty() && output_cells.back() == cell)
        {
            if (inspect_last && is_last)
            {
                inspect_flags.back() = 1;
            }
            continue;
        }

        output_cells.push_back(cell);
        inspect_flags.push_back((inspect_last && is_last) ? 1 : 0);
    }
}

PlannedPath planCoverage(
    int rows,
    int cols,
    const std::vector<GridCell>& blocked_cells,
    const std::vector<GridCell>& landing_candidates)
{
    PlannedPath plan;
    std::vector<std::vector<bool>> blocked(rows, std::vector<bool>(cols, false));
    for (const auto& cell : blocked_cells)
    {
        if (electronic_fly::isInsideGrid(cell, rows, cols))
        {
            blocked[cell.row][cell.col] = true;
        }
    }

    const std::vector<GridCell> targets = buildSurveyTargets(rows, cols, blocked);
    if (targets.empty())
    {
        ROS_WARN("path_planning: no reachable survey cells remain");
        return plan;
    }

    GridCell current = targets.front();
    appendPathSegment({current}, true, plan.survey_cells, plan.inspect_flags);

    for (const auto& target : targets)
    {
        const std::vector<GridCell> segment = bfsPath(current, target, rows, cols, blocked);
        if (segment.empty())
        {
            ROS_WARN_STREAM(
                "path_planning: failed to connect to target " << electronic_fly::buildGridCode(target.row, target.col));
            continue;
        }

        appendPathSegment(segment, true, plan.survey_cells, plan.inspect_flags);
        current = target;
    }

    int best_distance = std::numeric_limits<int>::max();
    for (const auto& candidate : landing_candidates)
    {
        if (!electronic_fly::isInsideGrid(candidate, rows, cols) || blocked[candidate.row][candidate.col])
        {
            continue;
        }

        const std::vector<GridCell> segment = bfsPath(current, candidate, rows, cols, blocked);
        if (segment.empty())
        {
            continue;
        }

        if (static_cast<int>(segment.size()) < best_distance)
        {
            best_distance = static_cast<int>(segment.size());
            plan.return_cells = segment;
        }
    }

    return plan;
}

void publishPathParams(ros::NodeHandle& nh, const PlannedPath& plan)
{
    std::vector<int> survey_rows;
    std::vector<int> survey_cols;
    for (const auto& cell : plan.survey_cells)
    {
        survey_rows.push_back(cell.row);
        survey_cols.push_back(cell.col);
    }

    std::vector<int> return_rows;
    std::vector<int> return_cols;
    for (const auto& cell : plan.return_cells)
    {
        return_rows.push_back(cell.row);
        return_cols.push_back(cell.col);
    }

    nh.setParam("/path_planning_x", survey_rows);
    nh.setParam("/path_planning_y", survey_cols);
    nh.setParam("/path_planning_rows", survey_rows);
    nh.setParam("/path_planning_cols", survey_cols);
    nh.setParam("/path_planning_inspect_flags", plan.inspect_flags);
    nh.setParam("/return_path_planning_x", return_rows);
    nh.setParam("/return_path_planning_y", return_cols);
    nh.setParam("/return_path_rows", return_rows);
    nh.setParam("/return_path_cols", return_cols);
    nh.setParam("/vision/path_planning_x", survey_rows);
    nh.setParam("/vision/path_planning_y", survey_cols);
    nh.setParam("/vision/path_planning_inspect_flags", plan.inspect_flags);
    nh.setParam("/vision/return_path_planning_x", return_rows);
    nh.setParam("/vision/return_path_planning_y", return_cols);
    nh.setParam("/path_planning_ready", !survey_rows.empty() && !survey_cols.empty());
}

std::string cellsToString(const std::vector<GridCell>& cells)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        if (index > 0)
        {
            output << ", ";
        }
        output << electronic_fly::buildGridCode(cells[index].row, cells[index].col);
    }
    return output.str();
}

}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "path_planning");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    setEmptyOutputs(nh);

    const int rows = nh.param("/MissionPlanner/grid_rows", pnh.param("grid_rows", 7));
    const int cols = nh.param("/MissionPlanner/grid_cols", pnh.param("grid_cols", 9));

    const std::vector<GridCell> blocked_cells = loadBlockedCells(nh, pnh, rows, cols);
    const std::vector<GridCell> landing_candidates = loadLandingCandidates(pnh, rows, cols);
    const PlannedPath plan = planCoverage(rows, cols, blocked_cells, landing_candidates);

    publishPathParams(nh, plan);

    ROS_INFO_STREAM(
        "path_planning: blocked=" << blocked_cells.size()
        << " survey_cells=" << plan.survey_cells.size()
        << " return_cells=" << plan.return_cells.size());
    if (!blocked_cells.empty())
    {
        ROS_INFO_STREAM("path_planning: blocked grid codes = " << cellsToString(blocked_cells));
    }

    ros::spinOnce();
    return 0;
}
