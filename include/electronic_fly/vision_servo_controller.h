// 功能：视觉伺服配置、状态和控制计算接口声明。

#pragma once

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/PositionTarget.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace electronic_fly
{

// 视觉伺服参数集合：只保存从 YAML 读取的参数，不保存运行时状态。
// mission_controller.cpp 每次悬停闭环时会读取这些参数，决定用比例模式还是 Jacobian 模式。
struct VisionServoConfig
{
    // 总开关与目标选择。
    bool enable = false;
    bool use_primary_target = true;

    // mode: "linear/proportional" 使用像素到米的经验比例；"jacobian" 使用图像雅可比反解。
    // control_mode: "position" 输出小位置修正；"velocity" 输出 XY 速度指令。
    // control_axis: "both/x/y" 用于单轴调试。
    std::string mode = "linear";
    std::string control_mode = "position";
    std::string control_axis = "both";
    bool use_yaw_compensation = true;

    // 目标像素不是默认图像中心，而是“激光理论落点”对应的图像偏移。
    double target_pixel_x = 0.0;
    double target_pixel_y = 0.0;

    // 比例模式使用：像素误差乘这个比例，近似得到飞机 XY 位移修正。
    double pixel_to_world_x = 0.0015;
    double pixel_to_world_y = 0.0015;

    // 根据高度把相机/激光安装偏移换算成像素偏移时使用，单位近似 px*m/m。
    double world_to_pixel_scale_x = 600.0;
    double world_to_pixel_scale_y = 600.0;

    // Jacobian 模式使用：J 表示飞机移动 1m 时，图像误差变化多少像素。
    double jacobian_sign_x = 1.0;
    double jacobian_sign_y = 1.0;
    double jacobian_jxx = 0.0;
    double jacobian_jxy = 0.0;
    double jacobian_jyx = 0.0;
    double jacobian_jyy = 0.0;
    double jacobian_damping = 30.0;

    double min_effective_altitude = 0.10;

    // P/D 增益与误差滤波。D 项不是飞控内环 D，而是视觉误差变化率的阻尼项。
    double kp_x = 1.0;
    double kp_y = 1.0;
    double kd_x = 0.0;
    double kd_y = 0.0;
    double error_filter_alpha = 0.0;

    // 输出限幅，防止单次位置修正或速度指令过大。
    double max_step_xy = 0.15;
    double max_velocity_xy = 0.10;
    double max_command_delta_xy = 0.0;

    // 检测跳变保护：YOLO 目标突然跳到另一个动物时，可以拒绝本次误差更新。
    double max_error_jump_px = 0.0;
    bool reject_error_jump = false;

    bool verbose_log = false;
    bool tune_log_enable = true;
    double tune_log_period = 0.25;

    // 死区与更新周期：误差小于死区时不继续追，避免飞机在目标附近来回抖。
    double deadband_px = 12.0;
    double update_period = 0.20;
    double max_total_offset_xy = 0.80;
    double settle_time = 1.5;

    // 速度控制下的“慢速接近 + 刹车保持”参数。
    // Speed-control hold/brake tuning.
    bool hold_brake_enable = true;
    double hold_release_px = 38.0;
    double hold_velocity_xy = 0.004;
    double velocity_damping_gain = 0.5;
    double velocity_damping_max_xy = 0.006;
    double brake_velocity_gain = 1.2;
    double brake_max_velocity_xy = 0.014;
    double brake_velocity_timeout = 1.2;
    double stable_speed_xy = 0.034;
    double stable_time = 0.6;

    // Center lock with short history and release hysteresis.
    bool center_lock_enable = true;
    double center_lock_time = 0.35;
    double center_lock_release_px = 40.0;
    double center_lock_velocity_xy = 0.010;
    double center_lock_pull_gain = 0.22;
    double center_lock_exit_speed_xy = 0.050;
    int center_lock_exit_speed_frames = 5;
    int center_lock_history_frames = 5;
    double center_lock_inner_pull_px = 14.0;
    double center_lock_inner_velocity_xy = 0.010;
    double center_lock_inner_pull_gain = 0.12;
    double center_lock_inner_brake_speed_xy = 0.040;

    // Continuous center damping and outer slowdown.
    bool center_continuous_enable = true;
    double center_continuous_radius_px = 40.0;
    double center_soft_error_px = 20.0;
    double center_soft_pull_gain = 0.28;
    double center_soft_velocity_xy = 0.010;
    double center_damping_gain = 1.10;
    double center_damping_max_xy = 0.018;
    double center_command_filter_alpha = 0.40;
    bool outer_slowdown_enable = true;
    double outer_slowdown_inner_px = 40.0;
    double outer_slowdown_outer_px = 90.0;
    double outer_slowdown_min_velocity_xy = 0.010;
    double outer_slowdown_damping_gain = 0.55;
    double outer_slowdown_damping_max_xy = 0.010;

    // Calibration mode for Jacobian estimation.
    double calibration_step_xy = 0.04;
    double calibration_hold_time = 3.0;

    double camera_offset_x = 0.0;
    double camera_offset_y = 0.0;
    double camera_offset_z = 0.0;
    double laser_offset_x = 0.0;
    double laser_offset_y = 0.0;
    double laser_offset_z = 0.0;
    bool log_enable = false;
    std::string log_path;
};

// Jacobian 标定的临时运行状态：记录每个标定位置的误差均值和实际飞机位置。
struct VisionServoCalibrationState
{
    bool active = false;
    bool complete = false;
    int phase = 0;
    ros::Time phase_start_time;
    std::vector<std::string> names;
    std::vector<double> offset_x;
    std::vector<double> offset_y;
    std::vector<double> sum_error_x;
    std::vector<double> sum_error_y;
    std::vector<double> sum_pose_x;
    std::vector<double> sum_pose_y;
    std::vector<int> samples;
};

struct VisionServoRuntimeState
{
    geometry_msgs::PoseStamped hover_anchor_pose;
    bool hover_anchor_valid = false;
    double servo_offset_x = 0.0;
    double servo_offset_y = 0.0;
    ros::Time last_servo_update_time;
    double servo_filtered_error_x_px = 0.0;
    double servo_filtered_error_y_px = 0.0;
    double servo_previous_error_x_px = 0.0;
    double servo_previous_error_y_px = 0.0;
    double servo_previous_raw_error_x_px = 0.0;
    double servo_previous_raw_error_y_px = 0.0;
    double servo_previous_command_x = 0.0;
    double servo_previous_command_y = 0.0;
    bool servo_history_valid = false;
    ros::Time vision_stable_since;
    bool vision_stable_logged = false;
    bool vision_center_lock_active = false;
    int vision_center_lock_speed_over_count = 0;
    std::deque<double> vision_center_lock_error_history_px;
    std::deque<double> vision_center_lock_speed_history_xy;
    bool velocity_servo_active = false;
    mavros_msgs::PositionTarget velocity_servo_target;
    VisionServoCalibrationState calibration;
};

struct VisionServoStepInput
{
    ros::Time now;
    double hover_elapsed = 0.0;
    double current_altitude = 0.0;
    double current_yaw = 0.0;
    double actual_vx = 0.0;
    double actual_vy = 0.0;
    double actual_speed_xy = 0.0;
    bool velocity_fresh = false;
    bool tracking_valid = false;
    double measured_dx = 0.0;
    double measured_dy = 0.0;
    geometry_msgs::PoseStamped pose_for_log;
};

struct VisionServoStepOutput
{
    std::string servo_phase = "hold";
    double error_x_px = 0.0;
    double error_y_px = 0.0;
    double error_norm_px = 0.0;
    double servo_error_x_px = 0.0;
    double servo_error_y_px = 0.0;
    double servo_filtered_error_x_px = 0.0;
    double servo_filtered_error_y_px = 0.0;
    double derivative_error_x_px = 0.0;
    double derivative_error_y_px = 0.0;
    double damped_error_x_px = 0.0;
    double damped_error_y_px = 0.0;
    double raw_command_x = 0.0;
    double raw_command_y = 0.0;
    double damping_command_x = 0.0;
    double damping_command_y = 0.0;
    double outer_slowdown_log_x = 0.0;
    double outer_slowdown_log_y = 0.0;
    double command_x = 0.0;
    double command_y = 0.0;
    double stable_elapsed = 0.0;
    double error_jump_px = 0.0;
    bool error_jump_rejected = false;
};

class VisionServoController
{
public:
    void resetRuntimeState(
        VisionServoRuntimeState& state,
        geometry_msgs::PoseStamped& target_pose,
        const geometry_msgs::PoseStamped& current_pose,
        bool hold_current_pose) const;

    void updateRuntime(
        const VisionServoConfig& config,
        bool vision_servo_velocity_control,
        const VisionServoStepInput& input,
        VisionServoRuntimeState& state,
        geometry_msgs::PoseStamped& target_pose,
        VisionServoStepOutput& output) const;
};

VisionServoConfig loadVisionServoConfig(ros::NodeHandle& pnh);

std::pair<double, double> computeLaserTargetPixelOffset(
    const VisionServoConfig& config,
    double current_altitude);

std::pair<double, double> computeVisionServoStep(
    const VisionServoConfig& config,
    double error_x_px,
    double error_y_px,
    double current_altitude,
    double current_yaw);

mavros_msgs::PositionTarget makeHoldPositionTarget(
    const geometry_msgs::PoseStamped& pose);

mavros_msgs::PositionTarget makeHoldPositionTarget(
    const geometry_msgs::PoseStamped& pose,
    double yaw);

mavros_msgs::PositionTarget makeVelocityTarget(
    const geometry_msgs::PoseStamped& hold_pose,
    double velocity_x,
    double velocity_y);

mavros_msgs::PositionTarget makeVelocityTarget(
    const geometry_msgs::PoseStamped& hold_pose,
    double velocity_x,
    double velocity_y,
    double yaw);

}  // namespace electronic_fly
