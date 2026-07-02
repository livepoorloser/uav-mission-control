// 功能：无人机任务主控制器，负责航点执行、视觉处理和降落流程。

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandLong.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/RCIn.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <XmlRpcValue.h>
#include "electronic_fly/detection_deduplicator.h"
#include "electronic_fly/detection_processor.h"
#include "electronic_fly/detection_types.h"
#include "electronic_fly/grid_detection_filter.h"
#include "electronic_fly/grid_projection_filter.h"
#include "electronic_fly/grid_utils.h"
#include "electronic_fly/json_utils.h"
#include "electronic_fly/trajectory_optimizer/trajectory_optimizer.h"
#include "electronic_fly/vision_servo_controller.h"

using electronic_fly::CameraProjectionRuntimeConfig;
using electronic_fly::DedupConfig;
using electronic_fly::DetectionDeduplicator;
using electronic_fly::DetectionFilterConfig;
using electronic_fly::DetectionSnapshot;
using electronic_fly::QuinticTrajectory;
using electronic_fly::TrajectoryOptimizerConfig;
using electronic_fly::TrajectorySample;
using electronic_fly::VisionServoCalibrationState;
using electronic_fly::VisionServoConfig;
using electronic_fly::Waypoint;
using electronic_fly::clusterSnapshotDetections;
using electronic_fly::computeLaserTargetPixelOffset;
using electronic_fly::computeVisionServoStep;
using electronic_fly::formatAcceptedDetectionsSummary;
using electronic_fly::formatCompetitionReport;
using electronic_fly::formatCountsSummary;
using electronic_fly::getDetectionSnapshot;
using electronic_fly::loadCameraProjectionConfig;
using electronic_fly::loadDedupConfig;
using electronic_fly::loadDetectionFilterConfig;
using electronic_fly::loadVisionServoConfig;
using electronic_fly::makeZeroCounts;
using electronic_fly::snapshotProjectedDetectionsToJson;
using electronic_fly::sumCounts;

std::mutex pose_mtx;
geometry_msgs::PoseStamped current_pose;
ros::Time last_pose_time;
bool have_pose = false;

std::mutex velocity_mtx;
geometry_msgs::TwistStamped current_velocity;
ros::Time last_velocity_time;
bool have_velocity = false;

mavros_msgs::State current_state;
bool rc_takeover = false;


//  用于锁定真实起飞点和起飞偏航角的全局变量
double start_x = 0.0;
double start_y = 0.0;
double start_z = 0.0;
double start_yaw = 0.0;
bool origin_locked = false; 

boost::property_tree::ptree latest_detection_summary;
bool have_detection_summary = false;

ros::Publisher setpoint_pub;
ros::Publisher raw_setpoint_pub;
ros::ServiceClient set_mode_client;
ros::ServiceClient arming_client;
ros::ServiceClient command_long_client;

struct FlightPlan{
    Waypoint takeoff_point;
    std::vector<Waypoint> waypoints;
    Waypoint land_point;
};

double clampValue(double value, double min_value, double max_value)
{
    return std::max(min_value, std::min(value, max_value));
}

enum FSM_State {
    WAIT_FOR_SETUP,
    TAKEOFF,
    HOVER_AT_TAKEOFF,
    WAYPOINT_FLIGHT,
    HOVER_AT_WAYPOINT,
    LANDING_APPROACH,
    AUTO_LAND_HANDOFF,
    AUTO_LAND,
    MISSION_COMPLETE,
};

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

void rc_cb(const mavros_msgs::RCIn::ConstPtr& msg){
    static bool initialized = false;
    static int last_ch5_value = 0;

    if(msg->channels.size() >= 5){
        int current_value = msg->channels[4];

        if(!initialized){
            last_ch5_value = current_value;
            initialized = true;
            ROS_INFO("RC channel 5 initialized: %d", current_value);
            return;
        }

        if(std::abs(current_value - last_ch5_value) > 200){
            ROS_ERROR("RC channel 5 changed from %d to %d, takeover detected! Exiting mission.", last_ch5_value, current_value);
            rc_takeover = true;
        }
        last_ch5_value = current_value;
    }
}

void vision_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
    std::lock_guard<std::mutex> lk(pose_mtx);
    current_pose = *msg;
    last_pose_time = ros::Time::now();
    have_pose = true;
    ROS_INFO_THROTTLE(1.0, "Received vision pose: (%.2f, %.2f, %.2f)", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void local_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
    std::lock_guard<std::mutex> lk(pose_mtx);
    current_pose = *msg;
    last_pose_time = ros::Time::now();
    have_pose = true;
    ROS_INFO_THROTTLE(1.0, "Received local pose: (%.2f, %.2f, %.2f)", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void odom_cb(const nav_msgs::Odometry::ConstPtr& msg){
    std::lock_guard<std::mutex> lk(pose_mtx);
    current_pose.header = msg->header;
    current_pose.pose = msg->pose.pose;
    last_pose_time = ros::Time::now();
    have_pose = true;
    ROS_INFO_THROTTLE(1.0, "Received odom pose: (%.2f, %.2f, %.2f)", msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
}

void velocity_cb(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lk(velocity_mtx);
    current_velocity = *msg;
    last_velocity_time = ros::Time::now();
    have_velocity = true;
}

void detection_cb(const std_msgs::String::ConstPtr& msg)
{
    boost::property_tree::ptree tree;
    if (electronic_fly::parseJson(msg->data, tree))
    {
        latest_detection_summary = tree;
        have_detection_summary = true;
    }
    else
    {
        ROS_WARN_THROTTLE(2.0, "mission_controller: failed to parse vision/detections_json");
    }
}

geometry_msgs::Quaternion yawToQuaternion(double yaw){
    geometry_msgs::Quaternion q;
    q.w = std::cos(yaw/2.0);
    q.x = 0;
    q.y = 0;
    q.z = std::sin(yaw/2.0);
    return q;
}
//四元数转成欧拉角 公式
double quaternionToYaw(const geometry_msgs::Quaternion& quaternion)
{
    return std::atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

void setTargetPoseFromWaypoint(geometry_msgs::PoseStamped& target_pose, const Waypoint& waypoint)
{   
    //先判断位置点是不是三个,不是直接返回
    if (waypoint.pos.size() != 3)
    {
        return;
    }
    //把三个xyz数字取出来 赋值
    target_pose.pose.position.x = waypoint.pos[0];
    target_pose.pose.position.y = waypoint.pos[1];
    target_pose.pose.position.z = waypoint.pos[2];
    target_pose.pose.orientation = yawToQuaternion(waypoint.yaw);
}
//轨迹生成  没啥用
void setTargetPoseFromSample(geometry_msgs::PoseStamped& target_pose, const TrajectorySample& sample)
{
    if (sample.pos.size() != 3)
    {
        return;
    }

    target_pose.pose.position.x = sample.pos[0];
    target_pose.pose.position.y = sample.pos[1];
    target_pose.pose.position.z = sample.pos[2];
    target_pose.pose.orientation = yawToQuaternion(sample.yaw);
}
//轨迹优化
TrajectoryOptimizerConfig loadTrajectoryOptimizerConfig(ros::NodeHandle& pnh)
{
    TrajectoryOptimizerConfig config;
    config.enable = pnh.param("TrajectoryOptimizer/enable", false);
    config.interpolate_yaw = pnh.param("TrajectoryOptimizer/interpolate_yaw", true);
    config.nominal_speed = pnh.param("TrajectoryOptimizer/nominal_speed", 0.8);
    config.max_acceleration = pnh.param("TrajectoryOptimizer/max_acceleration", 0.8);
    config.min_segment_time = pnh.param("TrajectoryOptimizer/min_segment_time", 1.0);
    config.max_segment_time = pnh.param("TrajectoryOptimizer/max_segment_time", 0.0);
    config.max_yaw_rate = pnh.param("TrajectoryOptimizer/max_yaw_rate", 0.8);
    config.max_tracking_error = pnh.param("TrajectoryOptimizer/max_tracking_error", 0.60);
    return config;
}

std::vector<std::string> loadClassNames(ros::NodeHandle& nh, ros::NodeHandle& pnh)
{
    std::vector<std::string> class_names;
    //三个if来确保读取到正确类别,后续可以优化
    if (!nh.getParam("/animal_detector/classes", class_names))
    {
        nh.getParam("animal_detector/classes", class_names);
    }
    if (class_names.empty())
    {
        pnh.getParam("Vision/classes", class_names);
    }
    if (class_names.empty())
    {
        class_names = {"tiger", "peacock", "monkey", "elephant", "wolf"};
    }
    return class_names;
}
//从 YAML 配置中读取任务规划参数
double getPlannerDoubleParam(ros::NodeHandle& pnh, const std::string& key, double default_value)
{
    return pnh.param("MissionPlanner/" + key, default_value);
}

bool getPlannerBoolParam(ros::NodeHandle& pnh, const std::string& key, bool default_value)
{
    return pnh.param("MissionPlanner/" + key, default_value);
}
//把path_planning 节点输出的网格坐标序列，转换成能直接执行的、带有真实 XY 坐标和悬停时间的航点列表
std::vector<Waypoint> buildPlannedWaypoints(
    ros::NodeHandle& nh,
    ros::NodeHandle& pnh,
    const Waypoint& takeoff_point,
    const Waypoint& land_point)
{
    std::vector<int> survey_rows;
    std::vector<int> survey_cols;
    if (!nh.getParam("/path_planning_rows", survey_rows) || !nh.getParam("/path_planning_cols", survey_cols) ||
        survey_rows.empty() || survey_rows.size() != survey_cols.size()) {
        return {};
    }

    std::vector<int> return_rows;
    std::vector<int> return_cols;
    nh.getParam("/return_path_rows", return_rows);
    nh.getParam("/return_path_cols", return_cols);
    std::vector<int> inspect_flags;
    nh.getParam("/path_planning_inspect_flags", inspect_flags);

    const double cell_size = getPlannerDoubleParam(pnh, "cell_size", 0.5);
    const double origin_x = getPlannerDoubleParam(pnh, "origin_x", 0.0);
    const double origin_y = getPlannerDoubleParam(pnh, "origin_y", 0.0);
    const double cruise_z = getPlannerDoubleParam(pnh, "takeoff_z", takeoff_point.pos[2]);
    const double row_direction = getPlannerDoubleParam(pnh, "row_direction", -1.0);
    const double survey_hover_time = getPlannerDoubleParam(pnh, "waypoint_hover_time", 2.0);
    const double return_hover_time = getPlannerDoubleParam(pnh, "return_hover_time", 0.0);
    const bool swap_axes = getPlannerBoolParam(pnh, "swap_grid_axes", true);
    const bool use_glide_landing = getPlannerBoolParam(pnh, "use_glide_landing", true);
    const double glide_slope_deg = getPlannerDoubleParam(pnh, "glide_slope_deg", 45.0);
    const double pi = std::acos(-1.0);
    const double glide_tan = std::tan(glide_slope_deg * pi / 180.0);
    const double safe_glide_tan = std::abs(glide_tan) < 1e-3 ? 1.0 : glide_tan;
    const double home_x = takeoff_point.pos[0];
    const double home_y = takeoff_point.pos[1];
    const double final_land_z = land_point.pos[2];

    std::vector<Waypoint> planned_waypoints;
    int last_row = std::numeric_limits<int>::min();
    int last_col = std::numeric_limits<int>::min();

    auto append_path =
        [&](const std::vector<int>& rows,
            const std::vector<int>& cols,
            const std::vector<int>* path_inspect_flags,
            bool default_inspect,
            double hover_time,
            bool apply_glide_profile) {
        if (rows.size() != cols.size()) {
            return;
        }

        for (std::size_t index = 0; index < rows.size(); ++index) {
            const int row = rows[index];
            const int col = cols[index];
            if (!planned_waypoints.empty() && row == last_row && col == last_col) {
                continue;
            }

            double x = 0.0;
            double y = 0.0;
            if (swap_axes) {
                x = origin_x + col * cell_size;
                y = origin_y + row * cell_size * row_direction;
            } else {
                x = origin_x + row * cell_size;
                y = origin_y + col * cell_size * row_direction;
            }

            double target_z = cruise_z;
            if (apply_glide_profile && use_glide_landing) {
                const double dx_home = x - home_x;
                const double dy_home = y - home_y;
                const double horizontal_distance = std::sqrt(dx_home * dx_home + dy_home * dy_home);
                target_z = final_land_z + horizontal_distance * safe_glide_tan;
                target_z = clampValue(target_z, final_land_z, cruise_z);
            }

            const bool inspect =
                path_inspect_flags && index < path_inspect_flags->size() ?
                    ((*path_inspect_flags)[index] != 0) :
                    default_inspect;
            const double effective_hover_time = inspect ? hover_time : 0.0;
            Waypoint waypoint;
            waypoint.pos = {x, y, target_z};
            waypoint.yaw = 0.0;
            waypoint.hover_time = effective_hover_time;
            waypoint.inspect = inspect;
            waypoint.grid_row = row;
            waypoint.grid_col = col;
            waypoint.grid_code = electronic_fly::buildGridCode(row, col);
            planned_waypoints.push_back(waypoint);
            last_row = row;
            last_col = col;
        }
    };

    append_path(survey_rows, survey_cols, &inspect_flags, true, survey_hover_time, false);
    append_path(return_rows, return_cols, nullptr, false, return_hover_time, true);

    return planned_waypoints;
}
//从 YAML 配置文件（或 ROS 参数服务器）中读取并构建完整的飞行计划，包括起飞点、所有巡检/返航航点、降落点
FlightPlan loadFlightPlanFromParams(ros::NodeHandle& nh, ros::NodeHandle& pnh)
{
    FlightPlan plan;
    XmlRpc::XmlRpcValue xml_waypoints;

    if (pnh.getParam("PathPoints/takeoff_point/pos", plan.takeoff_point.pos)) {
        pnh.getParam("PathPoints/takeoff_point/yaw", plan.takeoff_point.yaw);
        pnh.getParam("PathPoints/takeoff_point/hover_time", plan.takeoff_point.hover_time);
    } else {
        ROS_ERROR("Failed to load takeoff point. Using default [0, 0, 1.5].");
        plan.takeoff_point = {{0.0, 0.0, 1.5}, 0.0, 5.0, false};
    }

    if (pnh.getParam("PathPoints/land_point/pos", plan.land_point.pos)) {
        pnh.getParam("PathPoints/land_point/yaw", plan.land_point.yaw);
        pnh.getParam("PathPoints/land_point/hover_time", plan.land_point.hover_time);
    } else {
        ROS_WARN("Failed to load land point. Using default [0, 0, 0.0].");
        plan.land_point = {{0.0, 0.0, 0.0}, 0.0, 0.0, false};
    }

    const bool land_at_takeoff = getPlannerBoolParam(pnh, "land_at_takeoff", true);
    if (land_at_takeoff) {
        plan.land_point.pos[0] = plan.takeoff_point.pos[0];
        plan.land_point.pos[1] = plan.takeoff_point.pos[1];
    }

    const bool use_planned_path = getPlannerBoolParam(pnh, "use_planned_path", false);
    const double path_wait_timeout = getPlannerDoubleParam(pnh, "path_wait_timeout", 3.0);
    const bool require_complete_plan = getPlannerBoolParam(pnh, "require_complete_plan", true);
    if (use_planned_path) {
        const ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(path_wait_timeout);
        while (ros::ok() && ros::WallTime::now() < deadline) {
            if (require_complete_plan && !nh.param("/path_planning_ready", false)) {
                ros::WallDuration(0.1).sleep();
                continue;
            }

            plan.waypoints = buildPlannedWaypoints(nh, pnh, plan.takeoff_point, plan.land_point);
            if (!plan.waypoints.empty()) {
                ROS_INFO("Loaded %zu planned waypoints from /path_planning_rows and /path_planning_cols.", plan.waypoints.size());
                ROS_INFO("Flight Plan Loaded: Takeoff Z=%.2f, %zu waypoints.", plan.takeoff_point.pos[2], plan.waypoints.size());
                return plan;
            }
            ros::WallDuration(0.1).sleep();
        }
        ROS_WARN("MissionPlanner/use_planned_path enabled, but no planned waypoints were found. Falling back to YAML waypoints.");
    }

    if (pnh.getParam("PathPoints/waypoints", xml_waypoints) && xml_waypoints.getType() == XmlRpc::XmlRpcValue::TypeArray) {
        for (int i = 0; i < xml_waypoints.size(); ++i) {
            Waypoint wp;
            if (xml_waypoints[i].hasMember("pos") && xml_waypoints[i]["pos"].getType() == XmlRpc::XmlRpcValue::TypeArray) {
                for (int j = 0; j < xml_waypoints[i]["pos"].size(); ++j) {
                    if (xml_waypoints[i]["pos"][j].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
                        wp.pos.push_back(static_cast<double>(xml_waypoints[i]["pos"][j]));
                    } else if (xml_waypoints[i]["pos"][j].getType() == XmlRpc::XmlRpcValue::TypeInt) {
                        wp.pos.push_back(static_cast<double>(static_cast<int>(xml_waypoints[i]["pos"][j])));
                    } else {
                        ROS_WARN("Waypoint %d position element %d is not numeric. Skipping.", i, j);
                        wp.pos.clear();
                        break;
                    }
                }
            }
            if (xml_waypoints[i].hasMember("yaw")) {
                wp.yaw = static_cast<double>(xml_waypoints[i]["yaw"]);
            } else {
                wp.yaw = 0.0;
            }
            if (xml_waypoints[i].hasMember("hover_time")) {
                wp.hover_time = static_cast<double>(xml_waypoints[i]["hover_time"]);
            } else {
                wp.hover_time = 0.0;
            }
            wp.inspect = true;

            if (wp.pos.size() == 3) {
                const double cell_size = getPlannerDoubleParam(pnh, "cell_size", 0.5);
                const double origin_x = getPlannerDoubleParam(pnh, "origin_x", 0.0);
                const double origin_y = getPlannerDoubleParam(pnh, "origin_y", 0.0);
                const double row_direction = getPlannerDoubleParam(pnh, "row_direction", -1.0);
                const bool swap_axes = getPlannerBoolParam(pnh, "swap_grid_axes", true);
                if (cell_size > 1e-6 && std::abs(row_direction) > 1e-6)
                {
                    const double row_float = swap_axes
                        ? (wp.pos[1] - origin_y) / (cell_size * row_direction)
                        : (wp.pos[0] - origin_x) / cell_size;
                    const double col_float = swap_axes
                        ? (wp.pos[0] - origin_x) / cell_size
                        : (wp.pos[1] - origin_y) / (cell_size * row_direction);
                    wp.grid_row = static_cast<int>(std::lround(row_float));
                    wp.grid_col = static_cast<int>(std::lround(col_float));
                    wp.grid_code = electronic_fly::buildGridCode(wp.grid_row, wp.grid_col);
                }
                plan.waypoints.push_back(wp);
            } else {
                ROS_WARN("Waypoint %d position array is not size 3 or invalid data. Skipping.", i);
            }
        }
    } else {
        ROS_INFO("No waypoints loaded or 'PathPoints/waypoints' is missing/invalid.");
    }

    ROS_INFO("Flight Plan Loaded: Takeoff Z=%.2f, %zu waypoints.", plan.takeoff_point.pos[2], plan.waypoints.size());
    return plan;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mission_controller");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    FlightPlan plan;
    plan = loadFlightPlanFromParams(nh, pnh);
    //读取轨迹优化器的配置
    const TrajectoryOptimizerConfig trajectory_optimizer_config = loadTrajectoryOptimizerConfig(pnh);
    //读取视觉伺服的配置
    const VisionServoConfig vision_servo_config = loadVisionServoConfig(pnh);
    //是速度控制还是位置控制
    const bool vision_servo_velocity_control = vision_servo_config.control_mode == "velocity";
    if (vision_servo_config.control_mode != "position" && !vision_servo_velocity_control)
    {
        ROS_WARN(
            "Unknown VisionServo/control_mode='%s'; falling back to position setpoint servo.",
            vision_servo_config.control_mode.c_str());
    }
    //读取 检测过滤配置
    const DetectionFilterConfig detection_filter_config = loadDetectionFilterConfig(pnh);
    //读取相机标定参数
    CameraProjectionRuntimeConfig camera_projection_config = loadCameraProjectionConfig(pnh);
    //去重模块
    const DedupConfig dedup_config = loadDedupConfig(pnh);
    DetectionDeduplicator detection_deduplicator(dedup_config);
    std::ofstream vision_servo_log;
    //创建视觉伺服的csv文件
    if (vision_servo_config.log_enable)
    {
        vision_servo_log.open(vision_servo_config.log_path.c_str(), std::ios::out | std::ios::trunc);
        //csv要记录的信息
        if (vision_servo_log.is_open())
        {
            vision_servo_log
                << "time,hover_elapsed,label,score,area_px,"
                << "target_px_x,target_px_y,measured_dx,measured_dy,"
                << "error_x_px,error_y_px,servo_error_x_px,servo_error_y_px,"
                                << "filtered_error_x_px,filtered_error_y_px,"
                                << "derivative_error_x_px,derivative_error_y_px,"
                                << "damped_error_x_px,damped_error_y_px,"
                                << "raw_command_x,raw_command_y,damping_command_x,damping_command_y,"
                                << "error_jump_px,error_jump_rejected,"
                << "servo_phase,actual_vx,actual_vy,actual_speed_xy,stable_elapsed,"
                << "altitude,yaw,command_x,command_y,total_x,total_y,"
                << "pose_x,pose_y,pose_z,target_x,target_y,target_z,anchor_x,anchor_y\n";
            ROS_INFO_STREAM("vision_servo_log: writing CSV to " << vision_servo_config.log_path);
        }
        else
        {
            ROS_WARN_STREAM("vision_servo_log: failed to open " << vision_servo_config.log_path);
        }
    }
    //终端打印轨迹优化参数
    if (trajectory_optimizer_config.enable)
    {
        ROS_INFO(
            "Minimum-jerk return trajectory enabled: nominal_speed=%.2f m/s, max_acceleration=%.2f m/s^2, min_segment_time=%.2f s, max_yaw_rate=%.2f rad/s.",
            trajectory_optimizer_config.nominal_speed,
            trajectory_optimizer_config.max_acceleration,
            trajectory_optimizer_config.min_segment_time,
            trajectory_optimizer_config.max_yaw_rate);
    }
    else
    {
        ROS_INFO("Trajectory optimizer disabled.");
    }
    // 读取类名
    const std::vector<std::string> class_names = loadClassNames(nh, pnh);
    //初始化视觉参数
    nh.setParam("vision/current_target_grid", std::string()); //网格编码 
    nh.setParam("vision/mission_report_ready", false);//最终任务报告
    nh.setParam("vision/mission_status", std::string("pending"));//等待开始
    nh.setParam("vision/mission_summary_text", std::string());//文字摘要
    
    //飞行过程中是否开启视觉?
    const bool vision_enable_during_flight = pnh.param("Vision/enable_during_flight", false);
    //使用什么定位源?
    bool useVision = false, useLocal = false, useOdom = false;
    std::string subscribe_odometry_topic, subscribe_local_topic, subscribe_vision_topic, subscribe_velocity_topic;

    //读取定位参数
    pnh.param("VehiclePose/useVision", useVision, false);
    pnh.param("VehiclePose/useLocal", useLocal, false);
    pnh.param("VehiclePose/useOdometry", useOdom, false);
    pnh.param("VehiclePose/subscribe_Vision_topic", subscribe_vision_topic, std::string("mavros/vision_pose/pose"));
    pnh.param("VehiclePose/subscribe_Local_topic", subscribe_local_topic, std::string("mavros/local_position/pose"));
    pnh.param("VehiclePose/subscribe_odometry_topic", subscribe_odometry_topic, std::string("mavros/odometry/in"));
    pnh.param("VehiclePose/subscribe_velocity_topic", subscribe_velocity_topic, std::string("mavros/local_position/velocity_local"));
    //默认使用vision_pose   这块可以改进为默认使用更准的local_position 目前也是用local_position 进行定位
    ros::Subscriber pose_sub;
    if (useVision) {
        ROS_INFO("Subscribing to vision topic: %s", subscribe_vision_topic.c_str());
        pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(subscribe_vision_topic, 10, vision_cb);
    } else if (useLocal || (!useVision && !useOdom)) {
        useLocal = true;
        ROS_INFO("Subscribing to local topic: %s", subscribe_local_topic.c_str());
        pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(subscribe_local_topic, 10, local_cb);
    } else {
        useOdom = true;
        ROS_INFO("Subscribing to odom topic: %s", subscribe_odometry_topic.c_str());
        pose_sub = nh.subscribe<nav_msgs::Odometry>(subscribe_odometry_topic, 10, odom_cb);
    }
    //订阅飞控发来的速度数据
    ros::Subscriber velocity_sub = nh.subscribe<geometry_msgs::TwistStamped>(subscribe_velocity_topic, 10, velocity_cb);
    ROS_INFO("Subscribing to velocity topic: %s", subscribe_velocity_topic.c_str());
    //话题的订阅和发布
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    ros::Subscriber rc_sub = nh.subscribe<mavros_msgs::RCIn>("mavros/rc/in", 10, rc_cb);
    ros::Subscriber detection_sub = nh.subscribe<std_msgs::String>("vision/detections_json", 10, detection_cb);

    setpoint_pub = nh.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 20);
    raw_setpoint_pub = nh.advertise<mavros_msgs::PositionTarget>("mavros/setpoint_raw/local", 20);
    ros::Publisher vision_gate_pub = nh.advertise<std_msgs::Bool>("vision/check", 10);
    arming_client = nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
    command_long_client = nh.serviceClient<mavros_msgs::CommandLong>("mavros/cmd/command");
    set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

    ros::Rate rate(20.0);

    FSM_State exec_state = WAIT_FOR_SETUP;
    geometry_msgs::PoseStamped target_pose;
    geometry_msgs::PoseStamped auto_land_handoff_pose;
    ros::Time state_start_time;
    int current_waypoint_index = 0;
    std::map<std::string, int> mission_totals = makeZeroCounts(class_names);
    DetectionSnapshot waypoint_best_snapshot;
    waypoint_best_snapshot.counts = makeZeroCounts(class_names);
    double detection_window_start_sec = 0.0;
    double servo_offset_x = 0.0;
    double servo_offset_y = 0.0;
    ros::Time last_servo_update_time(0);
    double servo_filtered_error_x_px = 0.0;
    double servo_filtered_error_y_px = 0.0;
    double servo_previous_error_x_px = 0.0;
    double servo_previous_error_y_px = 0.0;
    double servo_previous_raw_error_x_px = 0.0;
    double servo_previous_raw_error_y_px = 0.0;
    double servo_previous_command_x = 0.0;
    double servo_previous_command_y = 0.0;
    bool servo_history_valid = false;
    ros::Time vision_stable_since(0);
    bool vision_stable_logged = false;
    bool velocity_servo_active = false;
    mavros_msgs::PositionTarget velocity_servo_target;
    geometry_msgs::PoseStamped hover_anchor_pose;
    bool hover_anchor_valid = false;
    VisionServoCalibrationState servo_calibration;
    QuinticTrajectory active_trajectory;
    ros::Time active_trajectory_start_time;
    ros::Time active_trajectory_last_update_time;
    double active_trajectory_elapsed = 0.0;
    bool trajectory_active = false;
    const double POSE_TIMEOUT = getPlannerDoubleParam(pnh, "pose_timeout", 2.0);
    const double ALT_TOLERANCE = getPlannerDoubleParam(pnh, "alt_tolerance", 0.18);
    const double POS_TOLERANCE = getPlannerDoubleParam(pnh, "pos_tolerance", 0.18);
    const double MAP_YAW_OFFSET_DEG = getPlannerDoubleParam(pnh, "map_yaw_offset_deg", 0.0);
    const double MAP_YAW_OFFSET_RAD = MAP_YAW_OFFSET_DEG * std::acos(-1.0) / 180.0;
    const double MAP_YAW_COS = std::cos(MAP_YAW_OFFSET_RAD);
    const double MAP_YAW_SIN = std::sin(MAP_YAW_OFFSET_RAD);
    const double grid_cell_size = getPlannerDoubleParam(pnh, "cell_size", 0.5);
    const double grid_origin_x = getPlannerDoubleParam(pnh, "origin_x", 0.0);
    const double grid_origin_y = getPlannerDoubleParam(pnh, "origin_y", 0.0);
    const double grid_row_direction = getPlannerDoubleParam(pnh, "row_direction", -1.0);
    const bool grid_swap_axes = getPlannerBoolParam(pnh, "swap_grid_axes", true);
    const int grid_rows = static_cast<int>(std::round(getPlannerDoubleParam(pnh, "grid_rows", 7.0)));
    const int grid_cols = static_cast<int>(std::round(getPlannerDoubleParam(pnh, "grid_cols", 9.0)));
    if (!pnh.hasParam("GridProjectionFilter/cell_size_x"))
    {
        camera_projection_config.roi.cell_size_x = grid_cell_size;
    }
    if (!pnh.hasParam("GridProjectionFilter/cell_size_y"))
    {
        camera_projection_config.roi.cell_size_y = grid_cell_size;
    }
    const double PRE_ARM_Z = 0.5;

    auto makeCurrentPoseWaypoint = [&]() -> Waypoint {
        std::lock_guard<std::mutex> lk(pose_mtx);
        Waypoint waypoint;
        waypoint.pos = {
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z};
        waypoint.yaw = quaternionToYaw(current_pose.pose.orientation);
        return waypoint;
    };

    auto mapToWorldPosition = [&](const std::vector<double>& map_position) -> std::vector<double> {
        if (map_position.size() != 3)
        {
            return map_position;
        }

        const double map_x = map_position[0];
        const double map_y = map_position[1];
        return {
            start_x + MAP_YAW_COS * map_x - MAP_YAW_SIN * map_y,
            start_y + MAP_YAW_SIN * map_x + MAP_YAW_COS * map_y,
            map_position[2]};
    };

    auto worldToMapPosition = [&](double world_x, double world_y, double world_z) -> std::vector<double> {
        const double dx = world_x - start_x;
        const double dy = world_y - start_y;
        return {
            MAP_YAW_COS * dx + MAP_YAW_SIN * dy,
            -MAP_YAW_SIN * dx + MAP_YAW_COS * dy,
            world_z};
    };

    auto gridCodeForWorldPoint = [&](const electronic_fly::Vec3& world_point, std::string& grid_code) -> bool {
        if (grid_cell_size <= 1e-6 || std::abs(grid_row_direction) <= 1e-6)
        {
            return false;
        }

        const auto map_position = worldToMapPosition(world_point.x, world_point.y, world_point.z);
        const double row_float = grid_swap_axes
            ? (map_position[1] - grid_origin_y) / (grid_cell_size * grid_row_direction)
            : (map_position[0] - grid_origin_x) / grid_cell_size;
        const double col_float = grid_swap_axes
            ? (map_position[0] - grid_origin_x) / grid_cell_size
            : (map_position[1] - grid_origin_y) / (grid_cell_size * grid_row_direction);

        electronic_fly::GridCell cell;
        cell.row = static_cast<int>(std::lround(row_float));
        cell.col = static_cast<int>(std::lround(col_float));
        if (!electronic_fly::isInsideGrid(cell, grid_rows, grid_cols))
        {
            return false;
        }

        grid_code = electronic_fly::buildGridCode(cell.row, cell.col);
        return true;
    };

    auto makeWorldWaypoint = [&](const Waypoint& relative_waypoint) -> Waypoint {
        Waypoint world_waypoint = relative_waypoint;
        if (relative_waypoint.pos.size() == 3)
        {
            world_waypoint.pos = mapToWorldPosition(relative_waypoint.pos);
        }
        world_waypoint.yaw = start_yaw + relative_waypoint.yaw;
        return world_waypoint;
    };

    auto startSegmentTo = [&](const Waypoint& relative_goal, const std::string& segment_name, bool use_minimum_jerk) {
        const Waypoint world_goal = makeWorldWaypoint(relative_goal);
        trajectory_active = false;

        if (trajectory_optimizer_config.enable && use_minimum_jerk)
        {
            const Waypoint start_waypoint = makeCurrentPoseWaypoint();
            active_trajectory = electronic_fly::buildTrajectory(
                start_waypoint,
                world_goal,
                trajectory_optimizer_config);
            if (active_trajectory.valid())
            {
                active_trajectory_start_time = ros::Time::now();
                active_trajectory_last_update_time = active_trajectory_start_time;
                active_trajectory_elapsed = 0.0;
                trajectory_active = true;
                setTargetPoseFromSample(target_pose, active_trajectory.sample(0.0));
                ROS_INFO(
                    "Minimum-jerk return trajectory: %s duration=%.2f s.",
                    segment_name.c_str(),
                    active_trajectory.duration());
                return;
            }

            ROS_WARN("Minimum-jerk return trajectory: failed to build %s trajectory, falling back to direct setpoint.", segment_name.c_str());
        }

        setTargetPoseFromWaypoint(target_pose, world_goal);
    };

    auto updateTrajectoryTarget = [&]() -> bool {
        if (!trajectory_active)
        {
            return true;
        }

        const ros::Time now = ros::Time::now();
        const double dt = std::max(0.0, (now - active_trajectory_last_update_time).toSec());
        active_trajectory_last_update_time = now;

        if (have_pose && trajectory_optimizer_config.max_tracking_error > 0.0)
        {
            geometry_msgs::PoseStamped pose_copy;
            {
                std::lock_guard<std::mutex> lk(pose_mtx);
                pose_copy = current_pose;
            }
            const double dx = pose_copy.pose.position.x - target_pose.pose.position.x;
            const double dy = pose_copy.pose.position.y - target_pose.pose.position.y;
            const double dz = pose_copy.pose.position.z - target_pose.pose.position.z;
            const double tracking_error = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (tracking_error > trajectory_optimizer_config.max_tracking_error)
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "trajectory_optimizer: holding trajectory progress, tracking_error=%.2f m.",
                    tracking_error);
                return false;
            }
        }

        active_trajectory_elapsed += dt;
        setTargetPoseFromSample(target_pose, active_trajectory.sample(active_trajectory_elapsed));
        return active_trajectory.isFinished(active_trajectory_elapsed);
    };

    auto resetVelocityServo = [&](bool hold_current_pose = false) {
        if (hold_current_pose && have_pose)
        {
            std::lock_guard<std::mutex> lk(pose_mtx);
            target_pose = current_pose;
        }
        velocity_servo_active = false;
        velocity_servo_target = electronic_fly::makeHoldPositionTarget(
            target_pose,
            quaternionToYaw(target_pose.pose.orientation));
        servo_filtered_error_x_px = 0.0;
        servo_filtered_error_y_px = 0.0;
        servo_previous_error_x_px = 0.0;
        servo_previous_error_y_px = 0.0;
        servo_previous_raw_error_x_px = 0.0;
        servo_previous_raw_error_y_px = 0.0;
        servo_previous_command_x = 0.0;
        servo_previous_command_y = 0.0;
        servo_history_valid = false;
        vision_stable_since = ros::Time(0);
        vision_stable_logged = false;
    };

    auto applyProjectionFilter = [&](const DetectionSnapshot& snapshot, const Waypoint& current_waypoint) -> DetectionSnapshot {
        if (!camera_projection_config.projection_enable ||
            snapshot.total <= 0 ||
            snapshot.detections.empty() ||
            current_waypoint.pos.size() != 3)
        {
            return snapshot;
        }

        geometry_msgs::PoseStamped pose_copy;
        {
            std::lock_guard<std::mutex> lk(pose_mtx);
            if (!have_pose)
            {
                return snapshot;
            }
            pose_copy = current_pose;
        }

        const std::vector<double> grid_center = mapToWorldPosition(current_waypoint.pos);
        if (grid_center.size() != 3)
        {
            return snapshot;
        }

        return electronic_fly::applyProjectionFilter(
            snapshot,
            current_waypoint,
            pose_copy,
            camera_projection_config,
            class_names,
            grid_center,
            gridCodeForWorldPoint,
            quaternionToYaw);
    };

    auto isDetectionAccepted = [&](const DetectionSnapshot& snapshot) -> bool {
        return electronic_fly::isDetectionAccepted(snapshot, detection_filter_config);
    };

    auto isDuplicateDetectionAtWithLimit = [&](
        const std::string& label,
        double world_x,
        double world_y,
        double world_z,
        const std::string& current_grid_code,
        std::size_t history_limit) -> bool {
        return detection_deduplicator.isDuplicateAtWithLimit(
            label,
            world_x,
            world_y,
            world_z,
            current_grid_code,
            history_limit);
    };

    auto isDuplicateDetectionAt = [&](
        const std::string& label,
        double world_x,
        double world_y,
        double world_z,
        const std::string& current_grid_code) -> bool {
        return isDuplicateDetectionAtWithLimit(
            label,
            world_x,
            world_y,
            world_z,
            current_grid_code,
            detection_deduplicator.historySize());
    };

    auto isDuplicateDetection = [&](const std::string& label, const Waypoint& waypoint) -> bool {
        const std::vector<double> world_position = mapToWorldPosition(waypoint.pos);
        if (world_position.size() != 3)
        {
            return false;
        }
        return isDuplicateDetectionAt(
            label,
            world_position[0],
            world_position[1],
            world_position[2],
            waypoint.grid_code);
    };

    auto rememberAcceptedDetectionAt = [&](
        const std::string& label,
        double world_x,
        double world_y,
        double world_z,
        const std::string& grid_code) {
        detection_deduplicator.rememberAt(label, world_x, world_y, world_z, grid_code);
    };

    auto rememberAcceptedDetection = [&](const std::string& label, const Waypoint& waypoint) {
        const std::vector<double> world_position = mapToWorldPosition(waypoint.pos);
        if (world_position.size() != 3)
        {
            return;
        }
        rememberAcceptedDetectionAt(
            label,
            world_position[0],
            world_position[1],
            world_position[2],
            waypoint.grid_code);
    };

    auto publishMissionReport = [&](const std::string& status) -> std::string {
        std::ostringstream summary;
        summary << "status=" << status
                << " animal_total=" << sumCounts(mission_totals)
                << " breakdown=[" << formatCountsSummary(class_names, mission_totals, true) << "]"
                << " detections=["
                << formatAcceptedDetectionsSummary(detection_deduplicator.missionRecords())
                << "]";

        const std::string summary_text = summary.str();
        nh.setParam("vision/mission_status", status);
        nh.setParam("vision/mission_summary_text", summary_text);
        nh.setParam("vision/mission_report_ready", status == "complete");
        return summary_text;
    };

    ROS_INFO("Waiting for FCU connection and pose data...");
    ros::WallTime wait_start = ros::WallTime::now();
    while(ros::ok() && (!current_state.connected || !have_pose)){
        ros::spinOnce();
        rate.sleep();
        if ((ros::WallTime::now() - wait_start).toSec() > 10.0) {
            ROS_ERROR("Timeout waiting for FCU or Pose. Aborting.");
            return 1;
        }
    }
    ROS_INFO("FCU connected and pose received. Ready to start.");

    {
        std::lock_guard<std::mutex> lk(pose_mtx);
        target_pose = current_pose;
    }

    for(int i=0;i<100 && ros::ok();++i){
        target_pose.header.stamp = ros::Time::now();
        setpoint_pub.publish(target_pose);
        //起飞之前视觉暂停工作
        std_msgs::Bool gate_msg;
        gate_msg.data = false;
        vision_gate_pub.publish(gate_msg);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;
    mavros_msgs::CommandBool disarm_cmd;
    disarm_cmd.request.value = false;
    const bool use_px4_auto_land = getPlannerBoolParam(pnh, "use_px4_auto_land", false);
    const double offboard_land_hold_time = getPlannerDoubleParam(pnh, "offboard_land_hold_time", 0.8);
    const double offboard_land_xy_tolerance = getPlannerDoubleParam(pnh, "offboard_land_xy_tolerance", 0.18);
    const double offboard_land_z_tolerance = getPlannerDoubleParam(pnh, "offboard_land_z_tolerance", 0.08);
    const bool auto_land_after_glide = getPlannerBoolParam(pnh, "auto_land_after_glide", true);
    const double auto_land_handoff_height = getPlannerDoubleParam(pnh, "auto_land_handoff_height", 0.30);
    const double auto_land_yaw_hold_time = getPlannerDoubleParam(pnh, "auto_land_yaw_hold_time", 1.0);
    const bool force_disarm_after_landing = getPlannerBoolParam(pnh, "force_disarm_after_landing", false);
    const double force_disarm_delay = getPlannerDoubleParam(pnh, "force_disarm_delay", 0.0);
    const double normal_disarm_retry_period = getPlannerDoubleParam(pnh, "normal_disarm_retry_period", 0.30);
    const double force_disarm_retry_period = getPlannerDoubleParam(pnh, "force_disarm_retry_period", 0.20);
    bool px4_auto_land_active = use_px4_auto_land;
    bool auto_land_request_sent = false;
    bool auto_land_handoff_ready_latched = false;
    ros::Time auto_land_handoff_ready_time(0);
    bool offboard_landing_ready_latched = false;
    ros::Time offboard_landing_ready_time(0);
    ros::Time last_normal_disarm_request(0);
    ros::Time last_force_disarm_request(0);
    ros::Time last_request = ros::Time::now();


    auto checkReach = [&](const Waypoint& wp) -> bool {
        std::lock_guard<std::mutex> lk(pose_mtx);
        if (!have_pose) return false;

        // 目标点必须加上起飞时的初始物理偏移量
        const std::vector<double> world_position = mapToWorldPosition(wp.pos);
        if (world_position.size() != 3) return false;
        double target_x = world_position[0];
        double target_y = world_position[1];
        double target_z = world_position[2];

        double dx = current_pose.pose.position.x - target_x;
        double dy = current_pose.pose.position.y - target_y;
        double dz = current_pose.pose.position.z - target_z;
        double dist_sq = dx*dx + dy*dy;

        if (exec_state == TAKEOFF) {
            return std::fabs(dz) < ALT_TOLERANCE;
        }

        return (dist_sq < POS_TOLERANCE * POS_TOLERANCE) && (std::fabs(dz) < ALT_TOLERANCE);
    };

    auto checkOffboardLandingReady = [&]() -> bool {
        std::lock_guard<std::mutex> lk(pose_mtx);
        if (!have_pose) return false;

        const std::vector<double> world_position = mapToWorldPosition(plan.land_point.pos);
        if (world_position.size() != 3) return false;
        const double target_x = world_position[0];
        const double target_y = world_position[1];
        const double dx = current_pose.pose.position.x - target_x;
        const double dy = current_pose.pose.position.y - target_y;
        const double dist_xy = std::sqrt(dx * dx + dy * dy);
        const double relative_z = current_pose.pose.position.z - start_z;
        return dist_xy <= offboard_land_xy_tolerance &&
               relative_z <= plan.land_point.pos[2] + offboard_land_z_tolerance;
    };

    auto checkAutoLandHandoffReady = [&]() -> bool {
        std::lock_guard<std::mutex> lk(pose_mtx);
        if (!have_pose) return false;

        const std::vector<double> world_position = mapToWorldPosition(plan.land_point.pos);
        if (world_position.size() != 3) return false;
        const double dx = current_pose.pose.position.x - world_position[0];
        const double dy = current_pose.pose.position.y - world_position[1];
        const double dist_xy = std::sqrt(dx * dx + dy * dy);
        const double relative_z = current_pose.pose.position.z - start_z;
        return dist_xy <= offboard_land_xy_tolerance &&
               relative_z <= auto_land_handoff_height;
    };

    auto captureAutoLandHandoffPose = [&]() {
        std::lock_guard<std::mutex> lk(pose_mtx);
        auto_land_handoff_pose = current_pose;
        auto_land_handoff_pose.pose.orientation = current_pose.pose.orientation;
        target_pose = auto_land_handoff_pose;
    };

    auto requestLandingDisarm = [&]() {
        const ros::Time now = ros::Time::now();
        if ((now - last_normal_disarm_request).toSec() >= normal_disarm_retry_period)
        {
            if (arming_client.call(disarm_cmd))
            {
                if (disarm_cmd.response.success)
                {
                    ROS_INFO("FSM: Normal disarm request accepted.");
                }
                else
                {
                    ROS_WARN_THROTTLE(1.0, "FSM: Normal disarm rejected by FCU.");
                }
            }
            else
            {
                ROS_WARN_THROTTLE(1.0, "FSM: Failed to call normal disarm service.");
            }
            last_normal_disarm_request = now;
        }

        if (force_disarm_after_landing &&
            (now - offboard_landing_ready_time).toSec() >=
                offboard_land_hold_time + force_disarm_delay &&
            current_state.armed &&
            (now - last_force_disarm_request).toSec() >= force_disarm_retry_period)
        {
            mavros_msgs::CommandLong force_disarm_cmd;
            force_disarm_cmd.request.command = 400;  // MAV_CMD_COMPONENT_ARM_DISARM
            force_disarm_cmd.request.confirmation = 0;
            force_disarm_cmd.request.param1 = 0.0;     // disarm
            force_disarm_cmd.request.param2 = 21196.0; // PX4 force disarm magic value
            if (command_long_client.call(force_disarm_cmd) &&
                force_disarm_cmd.response.success)
            {
                ROS_WARN_THROTTLE(1.0, "FSM: Force disarm request accepted after landing.");
            }
            else
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "FSM: Force disarm request failed or rejected, result=%u.",
                    force_disarm_cmd.response.result);
            }
            last_force_disarm_request = now;
        }
    };

    while(ros::ok()){
        ros::spinOnce();

        std_msgs::Bool gate_msg;
        const bool waypoint_inspection_enabled =
            exec_state == HOVER_AT_WAYPOINT &&
            current_waypoint_index < static_cast<int>(plan.waypoints.size()) &&
            plan.waypoints[current_waypoint_index].inspect;
        const bool flight_vision_enabled =
            vision_enable_during_flight &&
            exec_state != WAIT_FOR_SETUP &&
            exec_state != AUTO_LAND &&
            exec_state != AUTO_LAND_HANDOFF &&
            exec_state != MISSION_COMPLETE;
        gate_msg.data = waypoint_inspection_enabled || flight_vision_enabled;
        vision_gate_pub.publish(gate_msg);

        if(rc_takeover){
            break;
        }

        if (have_pose && (ros::Time::now() - last_pose_time).toSec() > POSE_TIMEOUT &&
             (current_state.armed || current_state.mode == "OFFBOARD"))
        {
            ROS_ERROR("Pose timeout (%.1fs). Triggering AUTO.LAND.", (ros::Time::now() - last_pose_time).toSec());
            mavros_msgs::SetMode land_set_mode;
            land_set_mode.request.custom_mode = "AUTO.LAND";
            set_mode_client.call(land_set_mode);
            px4_auto_land_active = true;
            trajectory_active = false;
            exec_state = AUTO_LAND;
            last_request = ros::Time::now();
        }

        if(exec_state != AUTO_LAND && exec_state != AUTO_LAND_HANDOFF && exec_state != MISSION_COMPLETE && current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(1.0))){
            if(set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent){
                ROS_INFO("OFFBOARD mode requested.");
            }
            last_request = ros::Time::now();
        }else if(exec_state != AUTO_LAND && exec_state != AUTO_LAND_HANDOFF && exec_state != MISSION_COMPLETE && !current_state.armed && current_state.mode=="OFFBOARD" && (ros::Time::now()-last_request>ros::Duration(1.0))){
            if(arming_client.call(arm_cmd) && arm_cmd.response.success){
                ROS_INFO("Arming requested.");
            }
            last_request = ros::Time::now();
        }

        switch (exec_state) {
            case WAIT_FOR_SETUP:
                if (current_state.mode == "OFFBOARD" && current_state.armed) {
                    ROS_INFO("FSM: Setup complete. Starting Takeoff.");

                    if (!origin_locked) {
                        std::lock_guard<std::mutex> lk(pose_mtx);
                        start_x = current_pose.pose.position.x;
                        start_y = current_pose.pose.position.y;
                        start_z = current_pose.pose.position.z;
                        start_yaw = quaternionToYaw(current_pose.pose.orientation);
                        origin_locked = true;
                        ROS_INFO(
                            "LOCKED ORIGIN: X=%.2f, Y=%.2f, Z=%.2f, Yaw=%.2f rad, MapYawOffset=%.2f deg",
                            start_x,
                            start_y,
                            start_z,
                            start_yaw,
                            MAP_YAW_OFFSET_DEG);
                    }

                    startSegmentTo(plan.takeoff_point, "takeoff", false);

                    state_start_time = ros::Time::now();
                    exec_state = TAKEOFF;
                } else {
                    
                    if (have_pose) {
                        std::lock_guard<std::mutex> lk(pose_mtx);
                        target_pose.pose.position.x = current_pose.pose.position.x;
                        target_pose.pose.position.y = current_pose.pose.position.y;
                        target_pose.pose.position.z = current_pose.pose.position.z;
                        target_pose.pose.orientation = current_pose.pose.orientation;
                        target_pose.pose.position.z = current_pose.pose.position.z; 
                        target_pose.pose.orientation = current_pose.pose.orientation;
                    }
                }
                break;

            case TAKEOFF:
            {
                const bool trajectory_finished = updateTrajectoryTarget();
                if (trajectory_finished && checkReach(plan.takeoff_point)) {
                    trajectory_active = false;
                    ROS_INFO("FSM: Reached Takeoff point altitude (Z=%.2f). Hovering.", plan.takeoff_point.pos[2]);
                    state_start_time = ros::Time::now();
                    exec_state = HOVER_AT_TAKEOFF;
                }
                break;
            }

            case HOVER_AT_TAKEOFF:
                if ((ros::Time::now() - state_start_time).toSec() > plan.takeoff_point.hover_time) {
                    if (plan.waypoints.empty()) {
                        ROS_INFO("FSM: Takeoff hover complete. No waypoints found. Approaching landing point.");
                        exec_state = LANDING_APPROACH;

                     
                        startSegmentTo(plan.land_point, "landing approach", true);
                        break;
                    }

                    ROS_INFO("FSM: Takeoff hover complete. Moving to Waypoint 1.");
                    current_waypoint_index = 0;
                    exec_state = WAYPOINT_FLIGHT;

                    const auto& next_wp = plan.waypoints[current_waypoint_index];
                    nh.setParam("vision/current_target_grid", next_wp.grid_code);
                  
                    startSegmentTo(next_wp, "waypoint 1", false);
                }
                break;

            case WAYPOINT_FLIGHT:
            {
                const bool trajectory_finished = updateTrajectoryTarget();
                if (trajectory_finished && checkReach(plan.waypoints[current_waypoint_index])) {
                    trajectory_active = false;
                    if (plan.waypoints[current_waypoint_index].inspect) {
                        ROS_INFO("FSM: Reached Waypoint %d. Hovering for inspection.", current_waypoint_index + 1);
                    } else {
                        ROS_INFO("FSM: Reached Waypoint %d. Transit point.", current_waypoint_index + 1);
                    }
                    state_start_time = ros::Time::now();
                    detection_window_start_sec = state_start_time.toSec();
                    waypoint_best_snapshot = DetectionSnapshot{};
                    waypoint_best_snapshot.counts = makeZeroCounts(class_names);
                    have_detection_summary = false;
                    servo_offset_x = 0.0;
                    servo_offset_y = 0.0;
                    last_servo_update_time = ros::Time(0);
                    resetVelocityServo();
                    {
                        std::lock_guard<std::mutex> lk(pose_mtx);
                        hover_anchor_pose = target_pose;
                        hover_anchor_pose.pose.orientation = current_pose.pose.orientation;
                    }
                    hover_anchor_valid = true;
                    servo_calibration = VisionServoCalibrationState{};
                    if (vision_servo_config.enable && vision_servo_config.mode == "calibrate")
                    {
                        const double s = std::abs(vision_servo_config.calibration_step_xy);
                        servo_calibration.active = true;
                        servo_calibration.names = {"center", "plus_x", "minus_x", "plus_y", "minus_y"};
                        servo_calibration.offset_x = {0.0, s, -s, 0.0, 0.0};
                        servo_calibration.offset_y = {0.0, 0.0, 0.0, s, -s};
                        servo_calibration.sum_error_x.assign(servo_calibration.names.size(), 0.0);
                        servo_calibration.sum_error_y.assign(servo_calibration.names.size(), 0.0);
                        servo_calibration.sum_pose_x.assign(servo_calibration.names.size(), 0.0);
                        servo_calibration.sum_pose_y.assign(servo_calibration.names.size(), 0.0);
                        servo_calibration.samples.assign(servo_calibration.names.size(), 0);
                        servo_calibration.phase_start_time =
                            state_start_time + ros::Duration(vision_servo_config.settle_time);
                        ROS_INFO(
                            "vision_servo_calib: started step=%.3f settle=%.2f hold=%.2f anchor=(%.2f, %.2f)",
                            s,
                            vision_servo_config.settle_time,
                            vision_servo_config.calibration_hold_time,
                            hover_anchor_pose.pose.position.x,
                            hover_anchor_pose.pose.position.y);
                    }
                    exec_state = HOVER_AT_WAYPOINT;
                }
                break;
            }

            case HOVER_AT_WAYPOINT:
            {
                const bool inspect_enabled = plan.waypoints[current_waypoint_index].inspect;
                const DetectionSnapshot raw_snapshot =
                    inspect_enabled
                        ? getDetectionSnapshot(
                            nh,
                            latest_detection_summary,
                            have_detection_summary,
                            class_names,
                            detection_window_start_sec)
                        : DetectionSnapshot{};
                const DetectionSnapshot projected_snapshot =
                    inspect_enabled ? applyProjectionFilter(raw_snapshot, plan.waypoints[current_waypoint_index]) : raw_snapshot;
                if (inspect_enabled && raw_snapshot.total > 0)
                {
                    nh.setParam(
                        "vision/last_projected_detection_json",
                        snapshotProjectedDetectionsToJson(
                            projected_snapshot,
                            plan.waypoints[current_waypoint_index].grid_code));
                }
                const bool detection_accepted = isDetectionAccepted(projected_snapshot);
                const DetectionSnapshot snapshot =
                    detection_accepted
                        ? clusterSnapshotDetections(projected_snapshot, class_names, detection_filter_config)
                        : DetectionSnapshot{};
                if (raw_snapshot.total > 0 && !detection_accepted)
                {
                    ROS_INFO_THROTTLE(
                        1.0,
                        "mission_controller: filtered out detection dx=%.1f dy=%.1f score=%.2f area=%.1f projected_grid=%s projection=%s",
                        raw_snapshot.primary_dx,
                        raw_snapshot.primary_dy,
                        raw_snapshot.primary_score,
                        raw_snapshot.primary_area_px,
                        projected_snapshot.primary_grid_code.empty()
                            ? "unknown"
                            : projected_snapshot.primary_grid_code.c_str(),
                        projected_snapshot.primary_projection_valid ? "valid" : "invalid");
                }
                else if (raw_snapshot.total > 0 && projected_snapshot.total < raw_snapshot.total)
                {
                    ROS_INFO_THROTTLE(
                        1.0,
                        "mission_controller: grid projection kept %d/%d detections at waypoint %s",
                        projected_snapshot.total,
                        raw_snapshot.total,
                        plan.waypoints[current_waypoint_index].grid_code.c_str());
                }

                if (snapshot.total > 0 &&
                    (waypoint_best_snapshot.total <= 0 ||
                     snapshot.total > waypoint_best_snapshot.total ||
                     (snapshot.total == waypoint_best_snapshot.total &&
                       snapshot.timestamp >= waypoint_best_snapshot.timestamp))) {
                    waypoint_best_snapshot = snapshot;
                }

                const double hover_elapsed = (ros::Time::now() - state_start_time).toSec();
                if (servo_calibration.active &&
                    hover_anchor_valid &&
                    have_pose &&
                    !servo_calibration.complete)
                {
                    const ros::Time now = ros::Time::now();
                    if (servo_calibration.phase >= static_cast<int>(servo_calibration.names.size()))
                    {
                        servo_calibration.complete = true;
                    }
                    else
                    {
                        const int phase = servo_calibration.phase;
                        target_pose.pose.position.x =
                            hover_anchor_pose.pose.position.x + servo_calibration.offset_x[phase];
                        target_pose.pose.position.y =
                            hover_anchor_pose.pose.position.y + servo_calibration.offset_y[phase];
                        target_pose.pose.position.z = hover_anchor_pose.pose.position.z;
                        target_pose.pose.orientation = hover_anchor_pose.pose.orientation;

                        if (now < servo_calibration.phase_start_time)
                        {
                            ROS_INFO_THROTTLE(
                                0.5,
                                "vision_servo_calib: phase=%s settling target=(%.2f, %.2f)",
                                servo_calibration.names[phase].c_str(),
                                target_pose.pose.position.x,
                                target_pose.pose.position.y);
                        }
                        else if (snapshot.tracking_valid)
                        {
                            const double current_altitude = current_pose.pose.position.z;
                            const auto target_pixel =
                                computeLaserTargetPixelOffset(vision_servo_config, current_altitude);
                            const double measured_dx =
                                vision_servo_config.use_primary_target ? snapshot.primary_dx : snapshot.avg_dx;
                            const double measured_dy =
                                vision_servo_config.use_primary_target ? snapshot.primary_dy : snapshot.avg_dy;
                            const double error_x_px = measured_dx - target_pixel.first;
                            const double error_y_px = measured_dy - target_pixel.second;
                            servo_calibration.sum_error_x[phase] += error_x_px;
                            servo_calibration.sum_error_y[phase] += error_y_px;
                            servo_calibration.sum_pose_x[phase] += current_pose.pose.position.x;
                            servo_calibration.sum_pose_y[phase] += current_pose.pose.position.y;
                            servo_calibration.samples[phase] += 1;
                            ROS_INFO_THROTTLE(
                                0.5,
                                "vision_servo_calib: phase=%s sample=%d err=(%.1f, %.1f) pose=(%.2f, %.2f) target=(%.2f, %.2f)",
                                servo_calibration.names[phase].c_str(),
                                servo_calibration.samples[phase],
                                error_x_px,
                                error_y_px,
                                current_pose.pose.position.x,
                                current_pose.pose.position.y,
                                target_pose.pose.position.x,
                                target_pose.pose.position.y);
                        }

                        if (now >= servo_calibration.phase_start_time &&
                            (now - servo_calibration.phase_start_time).toSec() >=
                            vision_servo_config.calibration_hold_time)
                        {
                            double avg_x = std::numeric_limits<double>::quiet_NaN();
                            double avg_y = std::numeric_limits<double>::quiet_NaN();
                            double avg_pose_x = std::numeric_limits<double>::quiet_NaN();
                            double avg_pose_y = std::numeric_limits<double>::quiet_NaN();
                            if (servo_calibration.samples[phase] > 0)
                            {
                                avg_x =
                                    servo_calibration.sum_error_x[phase] /
                                    static_cast<double>(servo_calibration.samples[phase]);
                                avg_y =
                                    servo_calibration.sum_error_y[phase] /
                                    static_cast<double>(servo_calibration.samples[phase]);
                                avg_pose_x =
                                    servo_calibration.sum_pose_x[phase] /
                                    static_cast<double>(servo_calibration.samples[phase]);
                                avg_pose_y =
                                    servo_calibration.sum_pose_y[phase] /
                                    static_cast<double>(servo_calibration.samples[phase]);
                            }
                            ROS_INFO(
                                "vision_servo_calib: phase=%s done samples=%d avg_err=(%.2f, %.2f) avg_pose=(%.3f, %.3f) target_offset=(%.3f, %.3f)",
                                servo_calibration.names[phase].c_str(),
                                servo_calibration.samples[phase],
                                avg_x,
                                avg_y,
                                avg_pose_x,
                                avg_pose_y,
                                servo_calibration.offset_x[phase],
                                servo_calibration.offset_y[phase]);
                            servo_calibration.phase += 1;
                            servo_calibration.phase_start_time =
                                now + ros::Duration(vision_servo_config.settle_time);
                        }
                    }

                    if (!servo_calibration.complete &&
                        servo_calibration.phase >= static_cast<int>(servo_calibration.names.size()))
                    {
                        servo_calibration.complete = true;
                    }

                    if (servo_calibration.complete)
                    {
                        auto averageAt = [&](int index, bool x_axis) -> double {
                            if (index < 0 ||
                                index >= static_cast<int>(servo_calibration.samples.size()) ||
                                servo_calibration.samples[index] <= 0)
                            {
                                return std::numeric_limits<double>::quiet_NaN();
                            }
                            const double sum =
                                x_axis ? servo_calibration.sum_error_x[index]
                                       : servo_calibration.sum_error_y[index];
                            return sum / static_cast<double>(servo_calibration.samples[index]);
                        };
                        auto averagePoseAt = [&](int index, bool x_axis) -> double {
                            if (index < 0 ||
                                index >= static_cast<int>(servo_calibration.samples.size()) ||
                                servo_calibration.samples[index] <= 0)
                            {
                                return std::numeric_limits<double>::quiet_NaN();
                            }
                            const double sum =
                                x_axis ? servo_calibration.sum_pose_x[index]
                                       : servo_calibration.sum_pose_y[index];
                            return sum / static_cast<double>(servo_calibration.samples[index]);
                        };
                        const double center_x = averageAt(0, true);
                        const double center_y = averageAt(0, false);
                        const double plus_x_x = averageAt(1, true);
                        const double plus_x_y = averageAt(1, false);
                        const double minus_x_x = averageAt(2, true);
                        const double minus_x_y = averageAt(2, false);
                        const double plus_y_x = averageAt(3, true);
                        const double plus_y_y = averageAt(3, false);
                        const double minus_y_x = averageAt(4, true);
                        const double minus_y_y = averageAt(4, false);
                        const double center_pose_x = averagePoseAt(0, true);
                        const double center_pose_y = averagePoseAt(0, false);
                        const double plus_x_pose_x = averagePoseAt(1, true);
                        const double plus_x_pose_y = averagePoseAt(1, false);
                        const double minus_x_pose_x = averagePoseAt(2, true);
                        const double minus_x_pose_y = averagePoseAt(2, false);
                        const double plus_y_pose_x = averagePoseAt(3, true);
                        const double plus_y_pose_y = averagePoseAt(3, false);
                        const double minus_y_pose_x = averagePoseAt(4, true);
                        const double minus_y_pose_y = averagePoseAt(4, false);
                        const double x_axis_dx = plus_x_pose_x - minus_x_pose_x;
                        const double x_axis_dy = plus_x_pose_y - minus_x_pose_y;
                        const double y_axis_dx = plus_y_pose_x - minus_y_pose_x;
                        const double y_axis_dy = plus_y_pose_y - minus_y_pose_y;
                        const double det = x_axis_dx * y_axis_dy - y_axis_dx * x_axis_dy;
                        double jxx = std::numeric_limits<double>::quiet_NaN();
                        double jxy = std::numeric_limits<double>::quiet_NaN();
                        double jyx = std::numeric_limits<double>::quiet_NaN();
                        double jyy = std::numeric_limits<double>::quiet_NaN();
                        if (std::abs(det) > 1e-6)
                        {
                            const double error_axis_x_x = plus_x_x - minus_x_x;
                            const double error_axis_x_y = plus_x_y - minus_x_y;
                            const double error_axis_y_x = plus_y_x - minus_y_x;
                            const double error_axis_y_y = plus_y_y - minus_y_y;
                            jxx = (error_axis_x_x * y_axis_dy - error_axis_y_x * x_axis_dy) / det;
                            jxy = (-error_axis_x_x * y_axis_dx + error_axis_y_x * x_axis_dx) / det;
                            jyx = (error_axis_x_y * y_axis_dy - error_axis_y_y * x_axis_dy) / det;
                            jyy = (-error_axis_x_y * y_axis_dx + error_axis_y_y * x_axis_dx) / det;
                        }
                        else
                        {
                            ROS_WARN(
                                "vision_servo_calib: actual motion too small or collinear, det=%.6f. Calibration matrix invalid.",
                                det);
                        }
                        ROS_INFO(
                            "vision_servo_calib: actual center_pose=(%.3f, %.3f) x_axis=(%.3f, %.3f) y_axis=(%.3f, %.3f)",
                            center_pose_x,
                            center_pose_y,
                            x_axis_dx,
                            x_axis_dy,
                            y_axis_dx,
                            y_axis_dy);
                        ROS_INFO(
                            "vision_servo_calib: COMPLETE center=(%.2f, %.2f) Jxx=%.2f Jxy=%.2f Jyx=%.2f Jyy=%.2f px_per_m",
                            center_x,
                            center_y,
                            jxx,
                            jxy,
                            jyx,
                            jyy);
                        ROS_INFO(
                            "vision_servo_calib: yaml jacobian_jxx: %.2f jacobian_jxy: %.2f jacobian_jyx: %.2f jacobian_jyy: %.2f",
                            jxx,
                            jxy,
                            jyx,
                            jyy);
                        servo_calibration.active = false;
                    }
                }
                else if (vision_servo_config.enable &&
                    vision_servo_config.mode != "calibrate" &&
                    inspect_enabled &&
                    snapshot.tracking_valid &&
                    have_pose &&
                    hover_anchor_valid &&
                    hover_elapsed >= vision_servo_config.settle_time)
                {
                    const ros::Time now = ros::Time::now();
                    if ((now - last_servo_update_time).toSec() >= vision_servo_config.update_period)
                    {
                        double current_altitude = 0.0;
                        double current_yaw = 0.0;
                        double actual_vx = 0.0;
                        double actual_vy = 0.0;
                        double actual_speed_xy = 0.0;
                        bool velocity_fresh = false;
                        geometry_msgs::PoseStamped pose_for_log;
                        {
                            std::lock_guard<std::mutex> lk(pose_mtx);
                            pose_for_log = current_pose;
                            current_altitude = current_pose.pose.position.z;
                            current_yaw = quaternionToYaw(current_pose.pose.orientation);
                        }
                        {
                            std::lock_guard<std::mutex> lk(velocity_mtx);
                            velocity_fresh =
                                have_velocity &&
                                (now - last_velocity_time).toSec() <=
                                    std::max(0.05, vision_servo_config.brake_velocity_timeout);
                            if (velocity_fresh)
                            {
                                actual_vx = current_velocity.twist.linear.x;
                                actual_vy = current_velocity.twist.linear.y;
                                actual_speed_xy = std::hypot(actual_vx, actual_vy);
                            }
                        }
                        const auto target_pixel =
                            computeLaserTargetPixelOffset(vision_servo_config, current_altitude);
                        const double measured_dx =
                            vision_servo_config.use_primary_target ? snapshot.primary_dx : snapshot.avg_dx;
                        const double measured_dy =
                            vision_servo_config.use_primary_target ? snapshot.primary_dy : snapshot.avg_dy;
                        const double error_x_px = measured_dx - target_pixel.first;
                        const double error_y_px = measured_dy - target_pixel.second;
                        const double error_norm_px = std::hypot(error_x_px, error_y_px);
                        const bool outside_deadband =
                            std::abs(error_x_px) > vision_servo_config.deadband_px ||
                            std::abs(error_y_px) > vision_servo_config.deadband_px;

                        const double servo_error_x_px =
                            outside_deadband && std::abs(error_x_px) > vision_servo_config.deadband_px ? error_x_px : 0.0;
                        const double servo_error_y_px =
                            outside_deadband && std::abs(error_y_px) > vision_servo_config.deadband_px ? error_y_px : 0.0;
                        double command_x = 0.0;
                        double command_y = 0.0;
                        double raw_command_x = 0.0;
                        double raw_command_y = 0.0;
                        double damping_command_x = 0.0;
                        double damping_command_y = 0.0;
                        double derivative_error_x_px = 0.0;
                        double derivative_error_y_px = 0.0;
                        double damped_error_x_px = 0.0;
                        double damped_error_y_px = 0.0;
                        double error_jump_px = 0.0;
                        bool error_jump_rejected = false;
                        std::string servo_phase = "hold";
                        double stable_elapsed = 0.0;
                        if (outside_deadband)
                        {
                            servo_phase = "track";
                            vision_stable_since = ros::Time(0);
                            vision_stable_logged = false;
                            const double dt_servo = servo_history_valid
                                ? std::max(1e-3, (now - last_servo_update_time).toSec())
                                : std::max(1e-3, vision_servo_config.update_period);
                            const double alpha = vision_servo_config.error_filter_alpha;
                            if (!servo_history_valid)
                            {
                                servo_filtered_error_x_px = servo_error_x_px;
                                servo_filtered_error_y_px = servo_error_y_px;
                                servo_previous_error_x_px = servo_error_x_px;
                                servo_previous_error_y_px = servo_error_y_px;
                                servo_previous_raw_error_x_px = servo_error_x_px;
                                servo_previous_raw_error_y_px = servo_error_y_px;
                            }
                            else
                            {
                                error_jump_px = std::hypot(
                                    servo_error_x_px - servo_previous_raw_error_x_px,
                                    servo_error_y_px - servo_previous_raw_error_y_px);
                                const double max_error_jump =
                                    std::max(0.0, vision_servo_config.max_error_jump_px);
                                error_jump_rejected =
                                    vision_servo_config.reject_error_jump &&
                                    max_error_jump > 1e-9 &&
                                    error_jump_px > max_error_jump;
                                if (!error_jump_rejected)
                                {
                                    //滤波
                                    servo_filtered_error_x_px =
                                        alpha * servo_filtered_error_x_px + (1.0 - alpha) * servo_error_x_px;
                                    servo_filtered_error_y_px =
                                        alpha * servo_filtered_error_y_px + (1.0 - alpha) * servo_error_y_px;
                                    servo_previous_raw_error_x_px = servo_error_x_px;
                                    servo_previous_raw_error_y_px = servo_error_y_px;
                                }
                            }

                            derivative_error_x_px =
                                (servo_filtered_error_x_px - servo_previous_error_x_px) / dt_servo;
                            derivative_error_y_px =
                                (servo_filtered_error_y_px - servo_previous_error_y_px) / dt_servo;
                            damped_error_x_px =
                                servo_filtered_error_x_px + vision_servo_config.kd_x * derivative_error_x_px;
                            damped_error_y_px =
                                servo_filtered_error_y_px + vision_servo_config.kd_y * derivative_error_y_px;
                            const auto servo_command = computeVisionServoStep(
                                vision_servo_config,
                                damped_error_x_px,
                                damped_error_y_px,
                                current_altitude,
                                current_yaw);
                            raw_command_x = servo_command.first;
                            raw_command_y = servo_command.second;
                            const double command_limit = vision_servo_velocity_control
                                ? vision_servo_config.max_velocity_xy
                                : vision_servo_config.max_step_xy;
                            command_x = clampValue(servo_command.first, -command_limit, command_limit);
                            command_y = clampValue(servo_command.second, -command_limit, command_limit);
                            const double command_delta_limit =
                                std::max(0.0, vision_servo_config.max_command_delta_xy);
                            if (servo_history_valid && command_delta_limit > 1e-9)
                            {
                                command_x = servo_previous_command_x + clampValue(
                                    command_x - servo_previous_command_x,
                                    -command_delta_limit,
                                    command_delta_limit);
                                command_y = servo_previous_command_y + clampValue(
                                    command_y - servo_previous_command_y,
                                    -command_delta_limit,
                                    command_delta_limit);
                            }
                            if (vision_servo_config.control_axis == "x")
                            {
                                command_y = 0.0;
                            }
                            else if (vision_servo_config.control_axis == "y")
                            {
                                command_x = 0.0;
                            }
                            const double hold_release_px = std::max(
                                vision_servo_config.deadband_px + 15.0,
                                vision_servo_config.hold_release_px);
                            const double hold_velocity_limit =
                                std::max(0.0, vision_servo_config.hold_velocity_xy);
                            if (vision_servo_config.hold_brake_enable &&
                                vision_servo_velocity_control &&
                                hold_velocity_limit > 1e-9 &&
                                error_norm_px <= hold_release_px)
                            {
                                servo_phase = "approach";
                                if (velocity_fresh)
                                {
                                    const double damping_limit =
                                        std::max(0.0, vision_servo_config.velocity_damping_max_xy);
                                    const double damping_gain =
                                        std::max(0.0, vision_servo_config.velocity_damping_gain);
                                    damping_command_x = clampValue(
                                        -damping_gain * actual_vx,
                                        -damping_limit,
                                        damping_limit);
                                    damping_command_y = clampValue(
                                        -damping_gain * actual_vy,
                                        -damping_limit,
                                        damping_limit);
                                    command_x += damping_command_x;
                                    command_y += damping_command_y;
                                }
                                command_x = clampValue(command_x, -hold_velocity_limit, hold_velocity_limit);
                                command_y = clampValue(command_y, -hold_velocity_limit, hold_velocity_limit);
                            }
                            servo_previous_error_x_px = servo_filtered_error_x_px;
                            servo_previous_error_y_px = servo_filtered_error_y_px;
                            servo_history_valid = true;
                        }
                        else
                        {
                            servo_history_valid = false;
                            const bool can_brake =
                                vision_servo_config.hold_brake_enable &&
                                vision_servo_velocity_control &&
                                velocity_fresh;
                            const bool still_moving =
                                can_brake &&
                                actual_speed_xy > std::max(0.0, vision_servo_config.stable_speed_xy);
                            if (still_moving)
                            {
                                servo_phase = "brake";
                                vision_stable_since = ros::Time(0);
                                vision_stable_logged = false;
                                const double brake_limit =
                                    std::max(0.0, vision_servo_config.brake_max_velocity_xy);
                                const double brake_gain =
                                    std::max(0.0, vision_servo_config.brake_velocity_gain);
                                command_x = clampValue(-brake_gain * actual_vx, -brake_limit, brake_limit);
                                command_y = clampValue(-brake_gain * actual_vy, -brake_limit, brake_limit);
                                raw_command_x = command_x;
                                raw_command_y = command_y;
                                if (vision_servo_config.control_axis == "x")
                                {
                                    command_y = 0.0;
                                }
                                else if (vision_servo_config.control_axis == "y")
                                {
                                    command_x = 0.0;
                                }
                            }
                            else
                            {
                                servo_phase = velocity_fresh ? "stable_hold" : "hold_no_velocity";
                                if (!velocity_fresh && vision_servo_config.hold_brake_enable)
                                {
                                    vision_stable_since = ros::Time(0);
                                    vision_stable_logged = false;
                                }
                                else if (vision_stable_since.isZero())
                                {
                                    vision_stable_since = now;
                                }
                                stable_elapsed = vision_stable_since.isZero()
                                    ? 0.0
                                    : (now - vision_stable_since).toSec();
                                if (!vision_stable_logged &&
                                    stable_elapsed >= std::max(0.0, vision_servo_config.stable_time))
                                {
                                    ROS_INFO(
                                        "vision_servo: stable target hold %.2fs err=(%.1f, %.1f) speed=%.3f m/s",
                                        stable_elapsed,
                                        error_x_px,
                                        error_y_px,
                                        actual_speed_xy);
                                    vision_stable_logged = true;
                                }
                            }
                        }

                        if (vision_servo_velocity_control)
                        {
                            const double anchor_delta_x =
                                pose_for_log.pose.position.x - hover_anchor_pose.pose.position.x;
                            const double anchor_delta_y =
                                pose_for_log.pose.position.y - hover_anchor_pose.pose.position.y;
                            const double max_offset = std::max(0.0, vision_servo_config.max_total_offset_xy);
                            if ((anchor_delta_x >= max_offset && command_x > 0.0) ||
                                (anchor_delta_x <= -max_offset && command_x < 0.0))
                            {
                                command_x = 0.0;
                            }
                            if ((anchor_delta_y >= max_offset && command_y > 0.0) ||
                                (anchor_delta_y <= -max_offset && command_y < 0.0))
                            {
                                command_y = 0.0;
                            }

                            velocity_servo_target = electronic_fly::makeVelocityTarget(
                                hover_anchor_pose,
                                command_x,
                                command_y,
                                quaternionToYaw(hover_anchor_pose.pose.orientation));
                            velocity_servo_active = true;
                            servo_offset_x = anchor_delta_x;
                            servo_offset_y = anchor_delta_y;
                        }
                        else if (outside_deadband)
                        {
                            const double next_servo_offset_x = clampValue(
                                servo_offset_x + command_x,
                                -vision_servo_config.max_total_offset_xy,
                                vision_servo_config.max_total_offset_xy);
                            const double next_servo_offset_y = clampValue(
                                servo_offset_y + command_y,
                                -vision_servo_config.max_total_offset_xy,
                                vision_servo_config.max_total_offset_xy);
                            command_x = next_servo_offset_x - servo_offset_x;
                            command_y = next_servo_offset_y - servo_offset_y;
                            servo_offset_x = next_servo_offset_x;
                            servo_offset_y = next_servo_offset_y;

                            resetVelocityServo();
                            std::lock_guard<std::mutex> lk(pose_mtx);
                            target_pose.pose.position.x = hover_anchor_pose.pose.position.x + servo_offset_x;
                            target_pose.pose.position.y = hover_anchor_pose.pose.position.y + servo_offset_y;
                            target_pose.pose.position.z = hover_anchor_pose.pose.position.z;
                            target_pose.pose.orientation = hover_anchor_pose.pose.orientation;
                        }
                        else
                        {
                            resetVelocityServo(true);
                        }

                        if (vision_servo_log.is_open())
                        {
                            vision_servo_log
                                << std::fixed << std::setprecision(6)
                                << now.toSec() << ','
                                << hover_elapsed << ','
                                << snapshot.primary_label << ','
                                << snapshot.primary_score << ','
                                << snapshot.primary_area_px << ','
                                << target_pixel.first << ','
                                << target_pixel.second << ','
                                << measured_dx << ','
                                << measured_dy << ','
                                << error_x_px << ','
                                << error_y_px << ','
                                << servo_error_x_px << ','
                                << servo_error_y_px << ','
                                << servo_filtered_error_x_px << ','
                                << servo_filtered_error_y_px << ','
                                << derivative_error_x_px << ','
                                << derivative_error_y_px << ','
                                << damped_error_x_px << ','
                                << damped_error_y_px << ','
                                << raw_command_x << ','
                                << raw_command_y << ','
                                << damping_command_x << ','
                                << damping_command_y << ','
                                << error_jump_px << ','
                                << (error_jump_rejected ? 1 : 0) << ','
                                << servo_phase << ','
                                << actual_vx << ','
                                << actual_vy << ','
                                << actual_speed_xy << ','
                                << stable_elapsed << ','
                                << current_altitude << ','
                                << current_yaw << ','
                                << command_x << ','
                                << command_y << ','
                                << servo_offset_x << ','
                                << servo_offset_y << ','
                                << pose_for_log.pose.position.x << ','
                                << pose_for_log.pose.position.y << ','
                                << pose_for_log.pose.position.z << ','
                                << target_pose.pose.position.x << ','
                                << target_pose.pose.position.y << ','
                                << target_pose.pose.position.z << ','
                                << hover_anchor_pose.pose.position.x << ','
                                << hover_anchor_pose.pose.position.y
                                << '\n';
                            vision_servo_log.flush();
                        }

                        if (vision_servo_config.tune_log_enable)
                        {
                            ROS_INFO_THROTTLE(
                                std::max(0.05, vision_servo_config.tune_log_period),
                                "VS_TUNE phase=%s in_db=%d fresh_v=%d err=(%.2f,%.2f|%.2f) "
                                "target=(%.2f,%.2f) meas=(%.2f,%.2f) "
                                "cmd=(%.5f,%.5f) raw=(%.5f,%.5f) vdamp=(%.5f,%.5f) "
                                "vel=(%.5f,%.5f|%.5f) offset=(%.3f,%.3f) stable=%.2f "
                                "limits(max_v=%.4f hold_v=%.4f brake_v=%.4f stable_v=%.4f deadband=%.1f)",
                                servo_phase.c_str(),
                                outside_deadband ? 0 : 1,
                                velocity_fresh ? 1 : 0,
                                error_x_px,
                                error_y_px,
                                error_norm_px,
                                target_pixel.first,
                                target_pixel.second,
                                measured_dx,
                                measured_dy,
                                command_x,
                                command_y,
                                raw_command_x,
                                raw_command_y,
                                damping_command_x,
                                damping_command_y,
                                actual_vx,
                                actual_vy,
                                actual_speed_xy,
                                servo_offset_x,
                                servo_offset_y,
                                stable_elapsed,
                                vision_servo_config.max_velocity_xy,
                                vision_servo_config.hold_velocity_xy,
                                vision_servo_config.brake_max_velocity_xy,
                                vision_servo_config.stable_speed_xy,
                                vision_servo_config.deadband_px);
                        }

                        if (vision_servo_config.verbose_log)
                        {
                            ROS_INFO_THROTTLE(
                                0.5,
                                "vision_servo[%s/%s/%s/%s]: target=(%.1f,%.1f) raw_err=(%.1f,%.1f) servo_err=(%.1f,%.1f) filt=(%.1f,%.1f) der=(%.1f,%.1f) damp=(%.1f,%.1f) raw_cmd=(%.3f,%.3f) v_damp=(%.3f,%.3f) cmd=(%.3f,%.3f) vel=(%.3f,%.3f|%.3f) stable=%.2f jump=%.1f%s alt=%.2f yaw=%.2f total=(%.3f,%.3f)",
                                vision_servo_config.mode.c_str(),
                                vision_servo_config.control_mode.c_str(),
                                vision_servo_config.control_axis.c_str(),
                                servo_phase.c_str(),
                                target_pixel.first,
                                target_pixel.second,
                                error_x_px,
                                error_y_px,
                                servo_error_x_px,
                                servo_error_y_px,
                                servo_filtered_error_x_px,
                                servo_filtered_error_y_px,
                                derivative_error_x_px,
                                derivative_error_y_px,
                                damped_error_x_px,
                                damped_error_y_px,
                                raw_command_x,
                                raw_command_y,
                                damping_command_x,
                                damping_command_y,
                                command_x,
                                command_y,
                                actual_vx,
                                actual_vy,
                                actual_speed_xy,
                                stable_elapsed,
                                error_jump_px,
                                error_jump_rejected ? " rejected" : "",
                                current_altitude,
                                current_yaw,
                                servo_offset_x,
                                servo_offset_y);
                        }
                        else
                        {
                            ROS_INFO_THROTTLE(
                                0.5,
                                "vision_servo[%s/%s/%s/%s]: target_px=(%.1f, %.1f) err_px=(%.1f, %.1f) servo_err=(%.1f, %.1f) vel=(%.3f, %.3f|%.3f) stable=%.2f alt=%.2f yaw=%.2f cmd=(%.3f, %.3f) total=(%.3f, %.3f) anchor=(%.2f, %.2f) target=(%.2f, %.2f)",
                                vision_servo_config.mode.c_str(),
                                vision_servo_config.control_mode.c_str(),
                                vision_servo_config.control_axis.c_str(),
                                servo_phase.c_str(),
                                target_pixel.first,
                                target_pixel.second,
                                error_x_px,
                                error_y_px,
                                servo_error_x_px,
                                servo_error_y_px,
                                actual_vx,
                                actual_vy,
                                actual_speed_xy,
                                stable_elapsed,
                                current_altitude,
                                current_yaw,
                                command_x,
                                command_y,
                                servo_offset_x,
                                servo_offset_y,
                                hover_anchor_pose.pose.position.x,
                                hover_anchor_pose.pose.position.y,
                                target_pose.pose.position.x,
                                target_pose.pose.position.y);
                        }
                        last_servo_update_time = now;
                        servo_previous_command_x = command_x;
                        servo_previous_command_y = command_y;
                    }
                }
                else if (vision_servo_velocity_control && velocity_servo_active)
                {
                    resetVelocityServo(true);
                }

                const bool waypoint_hover_complete =
                    (ros::Time::now() - state_start_time).toSec() > plan.waypoints[current_waypoint_index].hover_time;
                if (waypoint_hover_complete) {
                    if (inspect_enabled && waypoint_best_snapshot.total > 0) {
                        const Waypoint& current_waypoint = plan.waypoints[current_waypoint_index];
                        std::map<std::string, int> accepted_counts = makeZeroCounts(class_names);
                        const std::size_t history_limit_before_waypoint =
                            detection_deduplicator.historySize();
                        if (!waypoint_best_snapshot.detections.empty())
                        {
                            for (const auto& detection : waypoint_best_snapshot.detections)
                            {
                                std::vector<double> world_position;
                                std::string grid_code = current_waypoint.grid_code;
                                if (detection.projection_valid)
                                {
                                    world_position = {
                                        detection.world_x,
                                        detection.world_y,
                                        detection.world_z};
                                    if (!detection.projected_grid_code.empty())
                                    {
                                        grid_code = detection.projected_grid_code;
                                    }
                                }
                                else
                                {
                                    world_position = mapToWorldPosition(current_waypoint.pos);
                                }

                                if (world_position.size() != 3)
                                {
                                    continue;
                                }
                                if (isDuplicateDetectionAtWithLimit(
                                        detection.label,
                                        world_position[0],
                                        world_position[1],
                                        world_position[2],
                                        grid_code,
                                        history_limit_before_waypoint))
                                {
                                    continue;
                                }

                                mission_totals[detection.label] += 1;
                                accepted_counts[detection.label] += 1;
                                rememberAcceptedDetectionAt(
                                    detection.label,
                                    world_position[0],
                                    world_position[1],
                                    world_position[2],
                                    grid_code);
                            }
                        }
                        else
                        {
                            for (const auto& name : class_names) {
                                const int count = waypoint_best_snapshot.counts[name];
                                if (count <= 0) {
                                    continue;
                                }
                                if (isDuplicateDetection(name, current_waypoint)) {
                                    continue;
                                }
                                mission_totals[name] += count;
                                accepted_counts[name] += count;
                                for (int index = 0; index < count; ++index) {
                                    rememberAcceptedDetection(name, current_waypoint);
                                }
                            }
                        }
                        ROS_INFO_STREAM(
                            "mission_controller: waypoint " << current_waypoint_index + 1
                            << " animals=[" << formatCountsSummary(class_names, accepted_counts) << "]"
                            << " total=" << sumCounts(accepted_counts));
                    } else if (inspect_enabled) {
                        ROS_INFO_STREAM("mission_controller: waypoint " << current_waypoint_index + 1 << " animals=[none]");
                    }

                    current_waypoint_index++;
                    servo_offset_x = 0.0;
                    servo_offset_y = 0.0;
                    last_servo_update_time = ros::Time(0);
                    hover_anchor_valid = false;
                    resetVelocityServo(true);

                    if (current_waypoint_index < plan.waypoints.size()) {
                        ROS_INFO("FSM: Hover complete. Moving to Waypoint %d.", current_waypoint_index + 1);
                        exec_state = WAYPOINT_FLIGHT;

                        const auto& next_wp = plan.waypoints[current_waypoint_index];
                        nh.setParam("vision/current_target_grid", next_wp.grid_code);
                        startSegmentTo(next_wp, "waypoint transition", !next_wp.inspect);

                    } else {
                        ROS_INFO("FSM: All waypoints reached. Approaching landing point.");
                        exec_state = LANDING_APPROACH;
                        nh.setParam("vision/current_target_grid", std::string());
                        startSegmentTo(plan.land_point, "final landing approach", true);
                    }
                }
                break;
            }

            case LANDING_APPROACH:
            {
                const bool trajectory_finished = updateTrajectoryTarget();
                if (use_px4_auto_land && auto_land_after_glide && checkAutoLandHandoffReady())
                {
                    trajectory_active = false;
                    nh.setParam("vision/current_target_grid", std::string());
                    state_start_time = ros::Time::now();
                    if (auto_land_yaw_hold_time <= 1e-3)
                    {
                        px4_auto_land_active = true;
                        auto_land_request_sent = false;
                        last_request = ros::Time(0);
                        exec_state = AUTO_LAND;
                        ROS_INFO("FSM: Reached landing handoff height. Stopping setpoints and initiating PX4 AUTO.LAND.");
                        break;
                    }
                    captureAutoLandHandoffPose();
                    auto_land_handoff_ready_latched = true;
                    auto_land_handoff_ready_time = ros::Time::now();
                    exec_state = AUTO_LAND_HANDOFF;
                    ROS_INFO(
                        "FSM: Reached landing handoff height. Holding current yaw for %.2f s before PX4 AUTO.LAND.",
                        auto_land_yaw_hold_time);
                    break;
                }
                if (trajectory_finished && checkReach(plan.land_point)) {
                    trajectory_active = false;
                    setTargetPoseFromWaypoint(target_pose, makeWorldWaypoint(plan.land_point));
                    nh.setParam("vision/current_target_grid", std::string());
                    state_start_time = ros::Time::now();
                    if (use_px4_auto_land)
                    {
                        ROS_INFO("FSM: Reached landing point. Initiating PX4 AUTO.LAND.");
                        px4_auto_land_active = true;
                        auto_land_request_sent = false;
                        last_request = ros::Time(0);
                        exec_state = AUTO_LAND;
                    }
                    else
                    {
                        ROS_INFO("FSM: Reached landing point. Holding OFFBOARD low landing target.");
                        exec_state = AUTO_LAND;
                    }
                }
                break;
            }

            case AUTO_LAND_HANDOFF:
            {
                trajectory_active = false;
                target_pose = auto_land_handoff_pose;

                if (!auto_land_handoff_ready_latched)
                {
                    captureAutoLandHandoffPose();
                    auto_land_handoff_ready_latched = true;
                    auto_land_handoff_ready_time = ros::Time::now();
                    ROS_INFO(
                        "FSM: PX4 AUTO.LAND handoff height reached. Holding yaw for %.2f s.",
                        auto_land_yaw_hold_time);
                }

                if (auto_land_handoff_ready_latched &&
                    (ros::Time::now() - auto_land_handoff_ready_time).toSec() >= auto_land_yaw_hold_time)
                {
                    px4_auto_land_active = true;
                    auto_land_request_sent = false;
                    last_request = ros::Time(0);
                    exec_state = AUTO_LAND;
                    ROS_INFO("FSM: Yaw hold complete. Initiating PX4 AUTO.LAND.");
                }
                break;
            }

            case AUTO_LAND:
                trajectory_active = false;
                if (px4_auto_land_active &&
                    current_state.mode != "AUTO.LAND" &&
                    (ros::Time::now() - last_request > ros::Duration(1.0))) {
                    mavros_msgs::SetMode land_set_mode;
                    land_set_mode.request.custom_mode = "AUTO.LAND";
                    if (set_mode_client.call(land_set_mode) && land_set_mode.response.mode_sent)
                    {
                        auto_land_request_sent = true;
                        ROS_INFO("FSM: Requesting AUTO.LAND mode.");
                    }
                    else if (!auto_land_request_sent)
                    {
                        ROS_WARN_THROTTLE(1.0, "FSM: AUTO.LAND mode request not accepted yet.");
                    }
                    last_request = ros::Time::now();
                }
                if (px4_auto_land_active)
                {
                    ROS_INFO_THROTTLE(
                        2.0,
                        "FSM: PX4 AUTO.LAND active. Waiting for FCU land detector and auto-disarm.");
                }
                if (!px4_auto_land_active)
                {
                    setTargetPoseFromWaypoint(target_pose, makeWorldWaypoint(plan.land_point));
                    const bool landing_ready = checkOffboardLandingReady();
                    if (landing_ready && !offboard_landing_ready_latched)
                    {
                        offboard_landing_ready_latched = true;
                        offboard_landing_ready_time = ros::Time::now();
                        last_normal_disarm_request = ros::Time(0);
                        last_force_disarm_request = ros::Time(0);
                        ROS_INFO("FSM: OFFBOARD landing target reached. Preparing to disarm.");
                    }

                    if (offboard_landing_ready_latched &&
                        (ros::Time::now() - offboard_landing_ready_time).toSec() >= offboard_land_hold_time)
                    {
                        requestLandingDisarm();
                    }
                }
                if (!current_state.armed) {
                    ROS_INFO("FSM: Disarmed after landing. Mission complete.");
                    exec_state = MISSION_COMPLETE;
                }
                break;

            case MISSION_COMPLETE:
                if (!current_state.armed) {
                    std_msgs::Bool stop_gate_msg;
                    stop_gate_msg.data = false;
                    vision_gate_pub.publish(stop_gate_msg);
                    publishMissionReport("complete");
                    ROS_INFO_STREAM(formatCompetitionReport(
                        "complete",
                        class_names,
                        mission_totals,
                        detection_deduplicator.missionRecords()));
                    return 0;
                }
                break;
        }

        if ((!px4_auto_land_active || exec_state != AUTO_LAND) && exec_state != MISSION_COMPLETE) {
            const ros::Time publish_time = ros::Time::now();
            if (vision_servo_velocity_control && velocity_servo_active)
            {
                velocity_servo_target.header.stamp = publish_time;
                raw_setpoint_pub.publish(velocity_servo_target);
            }
            else
            {
                target_pose.header.stamp = publish_time;
                setpoint_pub.publish(target_pose);
            }
        }

        rate.sleep();
    }

    std_msgs::Bool stop_gate_msg;
    stop_gate_msg.data = false;
    vision_gate_pub.publish(stop_gate_msg);
    publishMissionReport("aborted");
    ROS_INFO_STREAM(formatCompetitionReport(
        "aborted",
        class_names,
        mission_totals,
        detection_deduplicator.missionRecords()));
    return 0;
}
