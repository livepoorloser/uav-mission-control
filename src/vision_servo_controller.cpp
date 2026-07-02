// 功能：视觉伺服控制器，将像素误差转换为位置或速度修正。

#include "electronic_fly/vision_servo_controller.h"

#include <algorithm>
#include <cmath>

namespace electronic_fly
{

namespace
{

double clampValue(double value, double min_value, double max_value)
{
    // 所有对飞机的修正量都要限幅，避免一次视觉误差造成过大的控制指令。
    return std::max(min_value, std::min(value, max_value));
}

double quaternionToYaw(const geometry_msgs::Quaternion& q)
{
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

mavros_msgs::PositionTarget makeHoldPositionTargetLocal(const geometry_msgs::PoseStamped& pose)
{
    mavros_msgs::PositionTarget target;
    target.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    target.type_mask =
        mavros_msgs::PositionTarget::IGNORE_VX |
        mavros_msgs::PositionTarget::IGNORE_VY |
        mavros_msgs::PositionTarget::IGNORE_VZ |
        mavros_msgs::PositionTarget::IGNORE_AFX |
        mavros_msgs::PositionTarget::IGNORE_AFY |
        mavros_msgs::PositionTarget::IGNORE_AFZ |
        mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    target.position.x = pose.pose.position.x;
    target.position.y = pose.pose.position.y;
    target.position.z = pose.pose.position.z;
    target.yaw = quaternionToYaw(pose.pose.orientation);
    return target;
}

mavros_msgs::PositionTarget makeVelocityTargetLocal(
    const geometry_msgs::PoseStamped& hold_pose,
    double velocity_x,
    double velocity_y)
{
    mavros_msgs::PositionTarget target;
    target.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    target.type_mask =
        mavros_msgs::PositionTarget::IGNORE_PX |
        mavros_msgs::PositionTarget::IGNORE_PY |
        mavros_msgs::PositionTarget::IGNORE_VZ |
        mavros_msgs::PositionTarget::IGNORE_AFX |
        mavros_msgs::PositionTarget::IGNORE_AFY |
        mavros_msgs::PositionTarget::IGNORE_AFZ |
        mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    target.position.z = hold_pose.pose.position.z;
    target.velocity.x = velocity_x;
    target.velocity.y = velocity_y;
    target.velocity.z = 0.0;
    target.yaw = quaternionToYaw(hold_pose.pose.orientation);
    return target;
}

double velocityCommandToPositionStep(const VisionServoConfig& config, double command)
{
    const double dt = std::max(1e-3, config.update_period);
    return command * dt;
}

double adaptVelocityFeedbackCommand(
    const VisionServoConfig& config,
    bool velocity_control,
    double velocity_command)
{
    if (velocity_control)
    {
        return velocity_command;
    }
    return velocityCommandToPositionStep(config, velocity_command);
}

}  // namespace

VisionServoConfig loadVisionServoConfig(ros::NodeHandle& pnh)
{
    // 统一从 YAML/ROS 参数服务器读取视觉伺服参数。
    // 主程序只拿到 VisionServoConfig，不直接散落读取几十个参数。
    VisionServoConfig config;
    config.enable = pnh.param("VisionServo/enable", false);
    config.use_primary_target = pnh.param("VisionServo/use_primary_target", true);
    config.mode = pnh.param<std::string>("VisionServo/mode", "jacobian");
    config.control_mode = pnh.param<std::string>("VisionServo/control_mode", "velocity");
    config.control_axis = pnh.param<std::string>("VisionServo/control_axis", "both");
    config.use_yaw_compensation = pnh.param("VisionServo/use_yaw_compensation", true);
    config.target_pixel_x = pnh.param("VisionServo/target_pixel_x", -9.5);
    config.target_pixel_y = pnh.param("VisionServo/target_pixel_y", 0.0);
    config.pixel_to_world_x = pnh.param("VisionServo/pixel_to_world_x", 0.002278);
    config.pixel_to_world_y = pnh.param("VisionServo/pixel_to_world_y", 0.002267);
    config.world_to_pixel_scale_x = pnh.param("VisionServo/world_to_pixel_scale_x", 439.02628474);
    config.world_to_pixel_scale_y = pnh.param("VisionServo/world_to_pixel_scale_y", 441.04615938);
    config.jacobian_sign_x = pnh.param("VisionServo/jacobian_sign_x", 1.0);
    config.jacobian_sign_y = pnh.param("VisionServo/jacobian_sign_y", -1.0);
    config.jacobian_jxx = pnh.param("VisionServo/jacobian_jxx", -61.78);
    config.jacobian_jxy = pnh.param("VisionServo/jacobian_jxy", 301.47);
    config.jacobian_jyx = pnh.param("VisionServo/jacobian_jyx", 220.72);
    config.jacobian_jyy = pnh.param("VisionServo/jacobian_jyy", -63.73);
    config.jacobian_damping = pnh.param("VisionServo/jacobian_damping", 150.0);
    config.min_effective_altitude = pnh.param("VisionServo/min_effective_altitude", 0.10);
    config.kp_x = pnh.param("VisionServo/kp_x", 0.10);
    config.kp_y = pnh.param("VisionServo/kp_y", 0.07);
    config.kd_x = pnh.param("VisionServo/kd_x", 0.04);
    config.kd_y = pnh.param("VisionServo/kd_y", 0.03);
    config.error_filter_alpha = clampValue(
        pnh.param("VisionServo/error_filter_alpha", 0.0),
        0.0,
        0.98);
    config.max_step_xy = pnh.param("VisionServo/max_step_xy", 0.005);
    config.max_velocity_xy = pnh.param("VisionServo/max_velocity_xy", 0.020);
    config.max_command_delta_xy = pnh.param("VisionServo/max_command_delta_xy", 0.006);
    config.max_error_jump_px = pnh.param("VisionServo/max_error_jump_px", 0.0);
    config.reject_error_jump = pnh.param("VisionServo/reject_error_jump", false);
    config.verbose_log = pnh.param("VisionServo/verbose_log", false);
    config.tune_log_enable = pnh.param("VisionServo/tune_log_enable", true);
    config.tune_log_period = pnh.param("VisionServo/tune_log_period", 0.25);
    config.deadband_px = pnh.param("VisionServo/deadband_px", 22.0);
    config.update_period = pnh.param("VisionServo/update_period", 0.20);
    config.max_total_offset_xy = pnh.param("VisionServo/max_total_offset_xy", 0.45);
    config.settle_time = pnh.param("VisionServo/settle_time", 1.5);
    config.hold_brake_enable = pnh.param("VisionServo/hold_brake_enable", true);
    config.hold_release_px = pnh.param("VisionServo/hold_release_px", 40.0);
    config.hold_velocity_xy = pnh.param("VisionServo/hold_velocity_xy", 0.010);
    config.velocity_damping_gain = pnh.param("VisionServo/velocity_damping_gain", 0.65);
    config.velocity_damping_max_xy = pnh.param("VisionServo/velocity_damping_max_xy", 0.010);
    config.brake_velocity_gain = pnh.param("VisionServo/brake_velocity_gain", 1.2);
    config.brake_max_velocity_xy = pnh.param("VisionServo/brake_max_velocity_xy", 0.014);
    config.brake_velocity_timeout = pnh.param("VisionServo/brake_velocity_timeout", 1.2);
    config.stable_speed_xy = pnh.param("VisionServo/stable_speed_xy", 0.034);
    config.stable_time = pnh.param("VisionServo/stable_time", 0.6);
    config.center_lock_enable = pnh.param("VisionServo/center_lock_enable", true);
    config.center_lock_time = pnh.param("VisionServo/center_lock_time", 0.35);
    config.center_lock_release_px = pnh.param("VisionServo/center_lock_release_px", 40.0);
    config.center_lock_velocity_xy = pnh.param("VisionServo/center_lock_velocity_xy", 0.010);
    config.center_lock_pull_gain = pnh.param("VisionServo/center_lock_pull_gain", 0.22);
    config.center_lock_exit_speed_xy = pnh.param("VisionServo/center_lock_exit_speed_xy", 0.050);
    config.center_lock_exit_speed_frames = pnh.param("VisionServo/center_lock_exit_speed_frames", 5);
    config.center_lock_history_frames = pnh.param("VisionServo/center_lock_history_frames", 5);
    config.center_lock_inner_pull_px = pnh.param("VisionServo/center_lock_inner_pull_px", 14.0);
    config.center_lock_inner_velocity_xy = pnh.param("VisionServo/center_lock_inner_velocity_xy", 0.010);
    config.center_lock_inner_pull_gain = pnh.param("VisionServo/center_lock_inner_pull_gain", 0.12);
    config.center_lock_inner_brake_speed_xy = pnh.param("VisionServo/center_lock_inner_brake_speed_xy", 0.040);
    config.center_continuous_enable = pnh.param("VisionServo/center_continuous_enable", true);
    config.center_continuous_radius_px = pnh.param("VisionServo/center_continuous_radius_px", 40.0);
    config.center_soft_error_px = pnh.param("VisionServo/center_soft_error_px", 20.0);
    config.center_soft_pull_gain = pnh.param("VisionServo/center_soft_pull_gain", 0.28);
    config.center_soft_velocity_xy = pnh.param("VisionServo/center_soft_velocity_xy", 0.010);
    config.center_damping_gain = pnh.param("VisionServo/center_damping_gain", 1.10);
    config.center_damping_max_xy = pnh.param("VisionServo/center_damping_max_xy", 0.018);
    config.center_command_filter_alpha = clampValue(
        pnh.param("VisionServo/center_command_filter_alpha", 0.40),
        0.0,
        0.98);
    config.outer_slowdown_enable = pnh.param("VisionServo/outer_slowdown_enable", true);
    config.outer_slowdown_inner_px = pnh.param("VisionServo/outer_slowdown_inner_px", 40.0);
    config.outer_slowdown_outer_px = pnh.param("VisionServo/outer_slowdown_outer_px", 90.0);
    config.outer_slowdown_min_velocity_xy = pnh.param("VisionServo/outer_slowdown_min_velocity_xy", 0.010);
    config.outer_slowdown_damping_gain = pnh.param("VisionServo/outer_slowdown_damping_gain", 0.55);
    config.outer_slowdown_damping_max_xy = pnh.param("VisionServo/outer_slowdown_damping_max_xy", 0.010);
    config.calibration_step_xy = pnh.param("VisionServo/calibration_step_xy", 0.04);
    config.calibration_hold_time = pnh.param("VisionServo/calibration_hold_time", 3.0);
    config.camera_offset_x = pnh.param("VisionServo/camera_offset_x", 0.0);
    config.camera_offset_y = pnh.param("VisionServo/camera_offset_y", 0.0);
    config.camera_offset_z = pnh.param("VisionServo/camera_offset_z", 0.0);
    config.laser_offset_x = pnh.param("VisionServo/laser_offset_x", 0.0);
    config.laser_offset_y = pnh.param("VisionServo/laser_offset_y", 0.0);
    config.laser_offset_z = pnh.param("VisionServo/laser_offset_z", 0.0);
    config.log_enable = pnh.param("VisionServo/log_enable", false);
    config.log_path = pnh.param<std::string>("VisionServo/log_path", "/tmp/vision_servo_log.csv");
    return config;
}

std::pair<double, double> computeLaserTargetPixelOffset(
    const VisionServoConfig& config,
    double current_altitude)
{
    // 相机中心看到的点，不一定等于激光真正会打到的位置。
    // 这里根据“激光 - 相机”的安装偏移和当前高度，估算激光落点在图像里的目标像素。
    const double relative_x = config.laser_offset_x - config.camera_offset_x;
    const double relative_y = config.laser_offset_y - config.camera_offset_y;
    const double relative_z = config.laser_offset_z - config.camera_offset_z;

    // 高度越低，同样的物理安装偏移在图像上表现出的像素偏移越大。
    // min_effective_altitude 用来防止高度过小导致除法放大。
    const double effective_altitude = std::max(
        config.min_effective_altitude,
        current_altitude - relative_z);
    const double target_x =
        config.target_pixel_x +
        (relative_x / effective_altitude) * config.world_to_pixel_scale_x;
    const double target_y =
        config.target_pixel_y +
        (relative_y / effective_altitude) * config.world_to_pixel_scale_y;
    return std::make_pair(target_x, target_y);
}

std::pair<double, double> computeVisionServoStep(
    const VisionServoConfig& config,
    double error_x_px,
    double error_y_px,
    double current_altitude,
    double current_yaw)
{
    if (config.mode != "jacobian")
    {
        // 简化比例模式：
        // 像素误差 e 乘经验比例 pixel_to_world，再乘 kp，得到飞机 XY 小修正。
        // 负号表示控制目标是让误差往 0 收敛。
        return std::make_pair(
            -error_x_px * config.pixel_to_world_x * config.kp_x,
            -error_y_px * config.pixel_to_world_y * config.kp_y);
    }

    // Jacobian 模式：
    // 先建立局部线性关系 e_new ~= e + J * dp。
    // 其中 e 是图像像素误差，dp 是飞机在 XY 平面的位移/速度指令方向。
    // 目标是找一个 dp，让 e + J * dp 尽量接近 0。
    const double relative_z = config.laser_offset_z - config.camera_offset_z;
    const double effective_altitude = std::max(
        config.min_effective_altitude,
        current_altitude - relative_z);

    // 如果 YAML 里已经填入实测 Jacobian，就优先用实测矩阵。
    // 否则退化为一个高度相关的对角近似：x 方向只影响图像 x，y 方向只影响图像 y。
    const bool use_calibrated_jacobian =
        std::abs(config.jacobian_jxx) > 1e-9 ||
        std::abs(config.jacobian_jxy) > 1e-9 ||
        std::abs(config.jacobian_jyx) > 1e-9 ||
        std::abs(config.jacobian_jyy) > 1e-9;
    double j00 = config.jacobian_jxx;
    double j01 = config.jacobian_jxy;
    double j10 = config.jacobian_jyx;
    double j11 = config.jacobian_jyy;
    if (!use_calibrated_jacobian)
    {
        j00 = config.jacobian_sign_x * config.world_to_pixel_scale_x / effective_altitude;
        j01 = 0.0;
        j10 = 0.0;
        j11 = config.jacobian_sign_y * config.world_to_pixel_scale_y / effective_altitude;
    }

    // 阻尼最小二乘反解：
    // dp = -(J^T J + lambda^2 I)^-1 J^T e
    // lambda 越大，解越保守；lambda 越小，越接近普通逆/伪逆。
    // 这里手写 2x2 矩阵求逆，避免额外依赖 Eigen。
    const double lambda = std::max(0.0, config.jacobian_damping);
    const double a00 = j00 * j00 + j10 * j10 + lambda * lambda;
    const double a01 = j00 * j01 + j10 * j11;
    const double a10 = j01 * j00 + j11 * j10;
    const double a11 = j01 * j01 + j11 * j11 + lambda * lambda;
    const double det = a00 * a11 - a01 * a10;
    if (std::abs(det) < 1e-9)
    {
        // 矩阵接近不可逆时不输出修正，避免产生异常大指令。
        return std::make_pair(0.0, 0.0);
    }

    const double inv00 = a11 / det;
    const double inv01 = -a01 / det;
    const double inv10 = -a10 / det;
    const double inv11 = a00 / det;
    const double jt_error_x = j00 * error_x_px + j10 * error_y_px;
    const double jt_error_y = j01 * error_x_px + j11 * error_y_px;

    // 这里得到的是机体系/局部 XY 下的修正量。
    double body_step_x = -config.kp_x * (inv00 * jt_error_x + inv01 * jt_error_y);
    double body_step_y = -config.kp_y * (inv10 * jt_error_x + inv11 * jt_error_y);

    if (use_calibrated_jacobian || !config.use_yaw_compensation)
    {
        return std::make_pair(body_step_x, body_step_y);
    }

    // 使用高度近似 Jacobian 时，它默认是在机体系下理解 x/y。
    // 如果实际向 MAVROS 发的是 local/world XY，则需要用当前 yaw 把机体系修正旋到世界系。
    const double cy = std::cos(current_yaw);
    const double sy = std::sin(current_yaw);
    return std::make_pair(
        cy * body_step_x - sy * body_step_y,
        sy * body_step_x + cy * body_step_y);
}

mavros_msgs::PositionTarget makeHoldPositionTarget(
    const geometry_msgs::PoseStamped& pose,
    double yaw)
{
    mavros_msgs::PositionTarget target;
    target.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    // 位置保持：使用 position.x/y/z 和 yaw，忽略速度、加速度、yaw_rate。
    target.type_mask =
        mavros_msgs::PositionTarget::IGNORE_VX |
        mavros_msgs::PositionTarget::IGNORE_VY |
        mavros_msgs::PositionTarget::IGNORE_VZ |
        mavros_msgs::PositionTarget::IGNORE_AFX |
        mavros_msgs::PositionTarget::IGNORE_AFY |
        mavros_msgs::PositionTarget::IGNORE_AFZ |
        mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    target.position.x = pose.pose.position.x;
    target.position.y = pose.pose.position.y;
    target.position.z = pose.pose.position.z;
    target.yaw = yaw;
    return target;
}

mavros_msgs::PositionTarget makeVelocityTarget(
    const geometry_msgs::PoseStamped& hold_pose,
    double velocity_x,
    double velocity_y,
    double yaw)
{
    mavros_msgs::PositionTarget target;
    target.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    // 速度闭环：只给 XY 速度，同时保留高度 z 和 yaw。
    // PX/PY 被忽略，避免位置 setpoint 和速度 setpoint 同时拉扯。
    target.type_mask =
        mavros_msgs::PositionTarget::IGNORE_PX |
        mavros_msgs::PositionTarget::IGNORE_PY |
        mavros_msgs::PositionTarget::IGNORE_VZ |
        mavros_msgs::PositionTarget::IGNORE_AFX |
        mavros_msgs::PositionTarget::IGNORE_AFY |
        mavros_msgs::PositionTarget::IGNORE_AFZ |
        mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    target.position.z = hold_pose.pose.position.z;
    target.velocity.x = velocity_x;
    target.velocity.y = velocity_y;
    target.velocity.z = 0.0;
    target.yaw = yaw;
    return target;
}

mavros_msgs::PositionTarget makeHoldPositionTarget(const geometry_msgs::PoseStamped& pose)
{
    return makeHoldPositionTargetLocal(pose);
}

mavros_msgs::PositionTarget makeVelocityTarget(
    const geometry_msgs::PoseStamped& hold_pose,
    double velocity_x,
    double velocity_y)
{
    return makeVelocityTargetLocal(hold_pose, velocity_x, velocity_y);
}

void VisionServoController::resetRuntimeState(
    VisionServoRuntimeState& state,
    geometry_msgs::PoseStamped& target_pose,
    const geometry_msgs::PoseStamped& current_pose,
    bool hold_current_pose) const
{
    if (hold_current_pose)
    {
        target_pose = current_pose;
    }

    state.velocity_servo_active = false;
    state.velocity_servo_target = makeHoldPositionTargetLocal(target_pose);
    state.servo_offset_x = 0.0;
    state.servo_offset_y = 0.0;
    state.last_servo_update_time = ros::Time(0);
    state.servo_filtered_error_x_px = 0.0;
    state.servo_filtered_error_y_px = 0.0;
    state.servo_previous_error_x_px = 0.0;
    state.servo_previous_error_y_px = 0.0;
    state.servo_previous_raw_error_x_px = 0.0;
    state.servo_previous_raw_error_y_px = 0.0;
    state.servo_previous_command_x = 0.0;
    state.servo_previous_command_y = 0.0;
    state.servo_history_valid = false;
    state.vision_stable_since = ros::Time(0);
    state.vision_stable_logged = false;
    state.vision_center_lock_active = false;
    state.vision_center_lock_speed_over_count = 0;
    state.vision_center_lock_error_history_px.clear();
    state.vision_center_lock_speed_history_xy.clear();
}

void VisionServoController::updateRuntime(
    const VisionServoConfig& config,
    bool vision_servo_velocity_control,
    const VisionServoStepInput& input,
    VisionServoRuntimeState& state,
    geometry_msgs::PoseStamped& target_pose,
    VisionServoStepOutput& output) const
{
    output = VisionServoStepOutput{};
    output.servo_phase = "hold";

    if (!input.tracking_valid || !config.enable)
    {
        state.velocity_servo_active = false;
        state.velocity_servo_target = makeHoldPositionTargetLocal(target_pose);
        state.servo_history_valid = false;
        state.vision_stable_since = ros::Time(0);
        state.vision_stable_logged = false;
        state.vision_center_lock_active = false;
        state.vision_center_lock_speed_over_count = 0;
        state.vision_center_lock_error_history_px.clear();
        state.vision_center_lock_speed_history_xy.clear();
        return;
    }

    const auto target_pixel = computeLaserTargetPixelOffset(config, input.current_altitude);
    const double error_x_px = input.measured_dx - target_pixel.first;
    const double error_y_px = input.measured_dy - target_pixel.second;
    const double error_norm_px = std::hypot(error_x_px, error_y_px);
    const bool outside_deadband =
        std::abs(error_x_px) > config.deadband_px ||
        std::abs(error_y_px) > config.deadband_px;

    const double servo_error_x_px =
        outside_deadband && std::abs(error_x_px) > config.deadband_px ? error_x_px : 0.0;
    const double servo_error_y_px =
        outside_deadband && std::abs(error_y_px) > config.deadband_px ? error_y_px : 0.0;

    output.error_x_px = error_x_px;
    output.error_y_px = error_y_px;
    output.error_norm_px = error_norm_px;
    output.servo_error_x_px = servo_error_x_px;
    output.servo_error_y_px = servo_error_y_px;

    double command_x = 0.0;
    double command_y = 0.0;
    double raw_command_x = 0.0;
    double raw_command_y = 0.0;
    double damping_command_x = 0.0;
    double damping_command_y = 0.0;
    double outer_slowdown_log_x = 0.0;
    double outer_slowdown_log_y = 0.0;
    double derivative_error_x_px = 0.0;
    double derivative_error_y_px = 0.0;
    double damped_error_x_px = 0.0;
    double damped_error_y_px = 0.0;
    double error_jump_px = 0.0;
    bool error_jump_rejected = false;
    std::string servo_phase = "hold";

    const bool position_servo_with_velocity_feedback =
        !vision_servo_velocity_control && input.velocity_fresh;
    const bool servo_velocity_feedback_available =
        vision_servo_velocity_control || position_servo_with_velocity_feedback;
    const double hold_release_px = std::max(
        config.deadband_px + 15.0,
        config.hold_release_px);
    const double hold_velocity_limit = std::max(0.0, config.hold_velocity_xy);
    const bool can_brake =
        config.hold_brake_enable &&
        servo_velocity_feedback_available &&
        input.velocity_fresh;
    const double stable_speed_limit = std::max(0.0, config.stable_speed_xy);
    const bool still_moving =
        can_brake && input.actual_speed_xy > stable_speed_limit;
    const double center_lock_release_px = std::max(
        config.center_lock_release_px,
        config.deadband_px + 6.0);
    const double center_lock_time = std::max(0.0, config.center_lock_time);
    const bool center_lock_enabled =
        config.center_lock_enable &&
        config.hold_brake_enable &&
        servo_velocity_feedback_available &&
        center_lock_time > 1e-6;
    const double center_lock_exit_speed_limit = std::max(
        stable_speed_limit,
        std::max(0.0, config.center_lock_exit_speed_xy));
    const int center_lock_exit_speed_frames = std::max(
        1,
        config.center_lock_exit_speed_frames);
    if (!center_lock_enabled)
    {
        state.vision_center_lock_active = false;
        state.vision_center_lock_speed_over_count = 0;
        state.vision_center_lock_error_history_px.clear();
        state.vision_center_lock_speed_history_xy.clear();
    }
    if (state.vision_center_lock_active)
    {
        const bool center_lock_speed_fast =
            input.velocity_fresh &&
            input.actual_speed_xy > center_lock_exit_speed_limit;
        if (error_norm_px > center_lock_release_px)
        {
            state.vision_center_lock_active = false;
            state.vision_center_lock_speed_over_count = 0;
            state.vision_center_lock_error_history_px.clear();
            state.vision_center_lock_speed_history_xy.clear();
        }
        else if (center_lock_speed_fast)
        {
            state.vision_center_lock_speed_over_count++;
            if (state.vision_center_lock_speed_over_count >= center_lock_exit_speed_frames)
            {
                state.vision_center_lock_active = false;
                state.vision_center_lock_speed_over_count = 0;
                state.vision_center_lock_error_history_px.clear();
                state.vision_center_lock_speed_history_xy.clear();
            }
        }
        else
        {
            state.vision_center_lock_speed_over_count = 0;
        }
    }
    else
    {
        state.vision_center_lock_speed_over_count = 0;
        state.vision_center_lock_error_history_px.clear();
        state.vision_center_lock_speed_history_xy.clear();
    }
    const bool center_lock_holding =
        center_lock_enabled &&
        state.vision_center_lock_active &&
        outside_deadband &&
        error_norm_px <= center_lock_release_px;
    const double center_continuous_radius_px = std::max(
        config.deadband_px,
        config.center_continuous_radius_px);
    const bool center_continuous_active =
        config.center_continuous_enable &&
        config.hold_brake_enable &&
        servo_velocity_feedback_available &&
        input.velocity_fresh &&
        center_continuous_radius_px > 1e-6 &&
        error_norm_px <= center_continuous_radius_px;

    if (center_continuous_active)
    {
        state.servo_history_valid = false;
        state.vision_center_lock_active = false;
        state.vision_center_lock_speed_over_count = 0;
        state.vision_center_lock_error_history_px.clear();
        state.vision_center_lock_speed_history_xy.clear();

        state.servo_filtered_error_x_px = error_x_px;
        state.servo_filtered_error_y_px = error_y_px;
        damped_error_x_px = error_x_px;
        damped_error_y_px = error_y_px;

        const auto center_pull_command = computeVisionServoStep(
            config,
            error_x_px,
            error_y_px,
            input.current_altitude,
            input.current_yaw);
        const double soft_error_px = std::max(1.0, config.center_soft_error_px);
        const double soft_pull_gain = clampValue(config.center_soft_pull_gain, 0.0, 1.0);
        const double adaptive_pull_gain =
            soft_pull_gain * error_norm_px / (error_norm_px + soft_error_px);
        const double pull_limit = std::max(0.0, config.center_soft_velocity_xy);
        const double damping_gain = std::max(0.0, config.center_damping_gain);
        const double damping_limit = std::max(0.0, config.center_damping_max_xy);

        const double pull_command_x = clampValue(
            adaptive_pull_gain * center_pull_command.first,
            -pull_limit,
            pull_limit);
        const double pull_command_y = clampValue(
            adaptive_pull_gain * center_pull_command.second,
            -pull_limit,
            pull_limit);
        damping_command_x = clampValue(
            adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -damping_gain * input.actual_vx),
            -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit),
            adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit));
        damping_command_y = clampValue(
            adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -damping_gain * input.actual_vy),
            -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit),
            adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit));

        command_x = pull_command_x + damping_command_x;
        command_y = pull_command_y + damping_command_y;
        raw_command_x = command_x;
        raw_command_y = command_y;

        const double center_command_limit =
            std::max(pull_limit, adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit));
        if (center_command_limit > 1e-9)
        {
            command_x = clampValue(command_x, -center_command_limit, center_command_limit);
            command_y = clampValue(command_y, -center_command_limit, center_command_limit);
        }
        const double center_filter_alpha = config.center_command_filter_alpha;
        if (center_filter_alpha > 1e-6)
        {
            command_x =
                center_filter_alpha * state.servo_previous_command_x +
                (1.0 - center_filter_alpha) * command_x;
            command_y =
                center_filter_alpha * state.servo_previous_command_y +
                (1.0 - center_filter_alpha) * command_y;
        }
        if (center_command_limit > 1e-9)
        {
            command_x = clampValue(command_x, -center_command_limit, center_command_limit);
            command_y = clampValue(command_y, -center_command_limit, center_command_limit);
        }

        if (input.actual_speed_xy > stable_speed_limit)
        {
            servo_phase = "center_continuous_damp";
            state.vision_stable_since = ros::Time(0);
            state.vision_stable_logged = false;
        }
        else if (error_norm_px > config.deadband_px)
        {
            servo_phase = "center_continuous_pull";
            state.vision_stable_since = ros::Time(0);
            state.vision_stable_logged = false;
        }
        else
        {
            servo_phase = "center_continuous_hold";
            if (state.vision_stable_since.isZero())
            {
                state.vision_stable_since = input.now;
            }
            output.stable_elapsed = state.vision_stable_since.isZero()
                ? 0.0
                : (input.now - state.vision_stable_since).toSec();
            if (!state.vision_stable_logged &&
                output.stable_elapsed >= std::max(0.0, config.stable_time))
            {
                ROS_INFO(
                    "vision_servo: stable target hold %.2fs err=(%.1f, %.1f) speed=%.3f m/s",
                    output.stable_elapsed,
                    error_x_px,
                    error_y_px,
                    input.actual_speed_xy);
                state.vision_stable_logged = true;
            }
        }

        if (config.control_axis == "x")
        {
            command_y = 0.0;
        }
        else if (config.control_axis == "y")
        {
            command_x = 0.0;
        }
    }
    else if (center_lock_holding)
    {
        state.servo_history_valid = false;
        const double lock_velocity_limit =
            std::max(0.0, config.center_lock_velocity_xy);
        const double lock_pull_gain =
            clampValue(config.center_lock_pull_gain, 0.0, 1.0);
        const auto lock_pull_command = computeVisionServoStep(
            config,
            error_x_px,
            error_y_px,
            input.current_altitude,
            input.current_yaw);
        const double brake_limit = std::max(0.0, config.brake_max_velocity_xy);
        const double brake_gain = std::max(0.0, config.brake_velocity_gain);
        double lock_pull_x = 0.0;
        double lock_pull_y = 0.0;
        double lock_brake_x = 0.0;
        double lock_brake_y = 0.0;
        if (input.velocity_fresh)
        {
            lock_brake_x = clampValue(
                adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -brake_gain * input.actual_vx),
                -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit),
                adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit));
            lock_brake_y = clampValue(
                adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -brake_gain * input.actual_vy),
                -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit),
                adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit));
        }
        if (still_moving)
        {
            servo_phase = "lock_brake";
            state.vision_stable_since = ros::Time(0);
            state.vision_stable_logged = false;
            command_x = lock_brake_x;
            command_y = lock_brake_y;
        }
        else if (input.velocity_fresh)
        {
            servo_phase = "lock_hold";
            state.vision_stable_since = ros::Time(0);
            state.vision_stable_logged = false;
            lock_pull_x = lock_pull_gain * lock_pull_command.first;
            lock_pull_y = lock_pull_gain * lock_pull_command.second;
            command_x = lock_pull_x;
            command_y = lock_pull_y;
        }
        else
        {
            servo_phase = "lock_no_velocity";
            state.vision_stable_since = ros::Time(0);
            state.vision_stable_logged = false;
        }
        raw_command_x = command_x;
        raw_command_y = command_y;
        const double hold_lock_limit = lock_velocity_limit > 1e-9
            ? lock_velocity_limit
            : adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit);
        const double lock_limit = still_moving
            ? adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit)
            : hold_lock_limit;
        command_x = clampValue(command_x, -lock_limit, lock_limit);
        command_y = clampValue(command_y, -lock_limit, lock_limit);
        if (config.control_axis == "x")
        {
            command_y = 0.0;
        }
        else if (config.control_axis == "y")
        {
            command_x = 0.0;
        }
    }
    else if (outside_deadband)
    {
        servo_phase = "track";
        state.vision_stable_since = ros::Time(0);
        state.vision_stable_logged = false;
        const double dt_servo = state.servo_history_valid
            ? std::max(1e-3, (input.now - state.last_servo_update_time).toSec())
            : std::max(1e-3, config.update_period);
        const double alpha = config.error_filter_alpha;
        if (!state.servo_history_valid)
        {
            state.servo_filtered_error_x_px = servo_error_x_px;
            state.servo_filtered_error_y_px = servo_error_y_px;
            state.servo_previous_error_x_px = servo_error_x_px;
            state.servo_previous_error_y_px = servo_error_y_px;
            state.servo_previous_raw_error_x_px = servo_error_x_px;
            state.servo_previous_raw_error_y_px = servo_error_y_px;
        }
        else
        {
            error_jump_px = std::hypot(
                servo_error_x_px - state.servo_previous_raw_error_x_px,
                servo_error_y_px - state.servo_previous_raw_error_y_px);
            const double max_error_jump = std::max(0.0, config.max_error_jump_px);
            error_jump_rejected =
                config.reject_error_jump &&
                max_error_jump > 1e-9 &&
                error_jump_px > max_error_jump;
            if (!error_jump_rejected)
            {
                state.servo_filtered_error_x_px =
                    alpha * state.servo_filtered_error_x_px + (1.0 - alpha) * servo_error_x_px;
                state.servo_filtered_error_y_px =
                    alpha * state.servo_filtered_error_y_px + (1.0 - alpha) * servo_error_y_px;
                state.servo_previous_raw_error_x_px = servo_error_x_px;
                state.servo_previous_raw_error_y_px = servo_error_y_px;
            }
        }

        derivative_error_x_px =
            (state.servo_filtered_error_x_px - state.servo_previous_error_x_px) / dt_servo;
        derivative_error_y_px =
            (state.servo_filtered_error_y_px - state.servo_previous_error_y_px) / dt_servo;
        damped_error_x_px =
            state.servo_filtered_error_x_px + config.kd_x * derivative_error_x_px;
        damped_error_y_px =
            state.servo_filtered_error_y_px + config.kd_y * derivative_error_y_px;
        const auto servo_command = computeVisionServoStep(
            config,
            damped_error_x_px,
            damped_error_y_px,
            input.current_altitude,
            input.current_yaw);
        raw_command_x = servo_command.first;
        raw_command_y = servo_command.second;
        const double command_limit = vision_servo_velocity_control
            ? config.max_velocity_xy
            : config.max_step_xy;
        double active_command_limit = command_limit;
        if (servo_velocity_feedback_available &&
            input.velocity_fresh &&
            config.outer_slowdown_enable)
        {
            const double slowdown_inner_px =
                std::max(config.center_continuous_radius_px, config.outer_slowdown_inner_px);
            const double slowdown_outer_px =
                std::max(slowdown_inner_px + 1.0, config.outer_slowdown_outer_px);
            if (error_norm_px > slowdown_inner_px &&
                error_norm_px <= slowdown_outer_px)
            {
                const double ring_alpha = clampValue(
                    (error_norm_px - slowdown_inner_px) /
                        (slowdown_outer_px - slowdown_inner_px),
                    0.0,
                    1.0);
                const double slow_limit =
                    clampValue(config.outer_slowdown_min_velocity_xy, 0.0, command_limit);
                active_command_limit = slow_limit + ring_alpha * (command_limit - slow_limit);
                const double slowdown_damping_limit =
                    std::max(0.0, config.outer_slowdown_damping_max_xy);
                const double slowdown_damping_gain =
                    std::max(0.0, config.outer_slowdown_damping_gain);
                outer_slowdown_log_x = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -slowdown_damping_gain * input.actual_vx),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, slowdown_damping_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, slowdown_damping_limit));
                outer_slowdown_log_y = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -slowdown_damping_gain * input.actual_vy),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, slowdown_damping_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, slowdown_damping_limit));
            }
        }
        command_x = clampValue(servo_command.first, -active_command_limit, active_command_limit);
        command_y = clampValue(servo_command.second, -active_command_limit, active_command_limit);
        command_x += outer_slowdown_log_x;
        command_y += outer_slowdown_log_y;
        command_x = clampValue(command_x, -active_command_limit, active_command_limit);
        command_y = clampValue(command_y, -active_command_limit, active_command_limit);
        const double command_delta_limit = std::max(0.0, config.max_command_delta_xy);
        if (state.servo_history_valid && command_delta_limit > 1e-9)
        {
            command_x = state.servo_previous_command_x + clampValue(
                command_x - state.servo_previous_command_x,
                -command_delta_limit,
                command_delta_limit);
            command_y = state.servo_previous_command_y + clampValue(
                command_y - state.servo_previous_command_y,
                -command_delta_limit,
                command_delta_limit);
        }
        if (config.control_axis == "x")
        {
            command_y = 0.0;
        }
        else if (config.control_axis == "y")
        {
            command_x = 0.0;
        }
        if (config.hold_brake_enable &&
            servo_velocity_feedback_available &&
            hold_velocity_limit > 1e-9 &&
            error_norm_px <= hold_release_px)
        {
            servo_phase = "approach";
            if (input.velocity_fresh)
            {
                const double damping_limit =
                    std::max(0.0, config.velocity_damping_max_xy);
                const double damping_gain =
                    std::max(0.0, config.velocity_damping_gain);
                damping_command_x = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -damping_gain * input.actual_vx),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit));
                damping_command_y = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -damping_gain * input.actual_vy),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, damping_limit));
                command_x += damping_command_x;
                command_y += damping_command_y;
            }
            command_x = clampValue(command_x, -hold_velocity_limit, hold_velocity_limit);
            command_y = clampValue(command_y, -hold_velocity_limit, hold_velocity_limit);
        }
        state.servo_previous_error_x_px = state.servo_filtered_error_x_px;
        state.servo_previous_error_y_px = state.servo_filtered_error_y_px;
        state.servo_history_valid = true;
    }
    else
    {
        state.servo_history_valid = false;
        const bool center_lock_centered =
            center_lock_enabled &&
            state.vision_center_lock_active &&
            !outside_deadband;
        if (center_lock_centered)
        {
            const size_t center_lock_history_limit = static_cast<size_t>(
                std::max(1, config.center_lock_history_frames));
            const double center_lock_inner_pull_px = std::max(
                0.0,
                config.center_lock_inner_pull_px);
            const double center_lock_inner_brake_speed_xy = std::max(
                0.0,
                config.center_lock_inner_brake_speed_xy);
            const double center_lock_inner_velocity_limit = std::max(
                0.0,
                config.center_lock_inner_velocity_xy);
            const double center_lock_inner_pull_gain = clampValue(
                config.center_lock_inner_pull_gain,
                0.0,
                1.0);
            const double brake_limit = std::max(0.0, config.brake_max_velocity_xy);
            const double brake_gain = std::max(0.0, config.brake_velocity_gain);
            double center_lock_pull_x = 0.0;
            double center_lock_pull_y = 0.0;
            double center_lock_brake_x = 0.0;
            double center_lock_brake_y = 0.0;
            state.vision_center_lock_error_history_px.push_back(error_norm_px);
            while (state.vision_center_lock_error_history_px.size() > center_lock_history_limit)
            {
                state.vision_center_lock_error_history_px.pop_front();
            }
            state.vision_center_lock_speed_history_xy.push_back(input.actual_speed_xy);
            while (state.vision_center_lock_speed_history_xy.size() > center_lock_history_limit)
            {
                state.vision_center_lock_speed_history_xy.pop_front();
            }
            const auto historyMean = [](const std::deque<double>& history) -> double {
                if (history.empty())
                {
                    return 0.0;
                }
                double sum = 0.0;
                for (const double value : history)
                {
                    sum += value;
                }
                return sum / static_cast<double>(history.size());
            };
            const double center_lock_error_mean_px =
                historyMean(state.vision_center_lock_error_history_px);
            const double center_lock_speed_mean_xy =
                historyMean(state.vision_center_lock_speed_history_xy);
            const bool center_lock_speed_fast_mean =
                input.velocity_fresh &&
                center_lock_speed_mean_xy > center_lock_inner_brake_speed_xy;
            const bool center_lock_needs_pull =
                center_lock_error_mean_px > center_lock_inner_pull_px;

            if (center_lock_speed_fast_mean)
            {
                servo_phase = "center_locked_brake";
                state.vision_stable_since = ros::Time(0);
                state.vision_stable_logged = false;
                center_lock_brake_x = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -brake_gain * input.actual_vx),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit));
                center_lock_brake_y = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -brake_gain * input.actual_vy),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit));
                command_x = center_lock_brake_x;
                command_y = center_lock_brake_y;
                raw_command_x = command_x;
                raw_command_y = command_y;
            }
            else if (center_lock_needs_pull)
            {
                servo_phase = "center_locked_pull";
                state.vision_stable_since = ros::Time(0);
                state.vision_stable_logged = false;
                const auto center_lock_pull_command = computeVisionServoStep(
                    config,
                    error_x_px,
                    error_y_px,
                    input.current_altitude,
                    input.current_yaw);
                center_lock_pull_x = center_lock_inner_pull_gain * center_lock_pull_command.first;
                center_lock_pull_y = center_lock_inner_pull_gain * center_lock_pull_command.second;
                command_x = clampValue(
                    center_lock_pull_x,
                    -center_lock_inner_velocity_limit,
                    center_lock_inner_velocity_limit);
                command_y = clampValue(
                    center_lock_pull_y,
                    -center_lock_inner_velocity_limit,
                    center_lock_inner_velocity_limit);
                raw_command_x = command_x;
                raw_command_y = command_y;
            }
            else
            {
                servo_phase = "center_locked";
                if (state.vision_stable_since.isZero())
                {
                    state.vision_stable_since = input.now;
                }
                output.stable_elapsed = state.vision_stable_since.isZero()
                    ? 0.0
                    : (input.now - state.vision_stable_since).toSec();
                if (!state.vision_stable_logged &&
                    output.stable_elapsed >= std::max(0.0, config.stable_time))
                {
                    ROS_INFO(
                        "vision_servo: stable target hold %.2fs err=(%.1f, %.1f) speed=%.3f m/s",
                        output.stable_elapsed,
                        error_x_px,
                        error_y_px,
                        input.actual_speed_xy);
                    state.vision_stable_logged = true;
                }
                raw_command_x = 0.0;
                raw_command_y = 0.0;
                command_x = 0.0;
                command_y = 0.0;
            }
            if (config.control_axis == "x")
            {
                command_y = 0.0;
            }
            else if (config.control_axis == "y")
            {
                command_x = 0.0;
            }
        }
        else
        {
            if (still_moving)
            {
                servo_phase = center_lock_enabled ? "capture_brake" : "brake";
                state.vision_center_lock_active = false;
                state.vision_center_lock_speed_over_count = 0;
                state.vision_stable_since = ros::Time(0);
                state.vision_stable_logged = false;
                const double brake_limit =
                    std::max(0.0, config.brake_max_velocity_xy);
                const double brake_gain =
                    std::max(0.0, config.brake_velocity_gain);
                command_x = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -brake_gain * input.actual_vx),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit));
                command_y = clampValue(
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, -brake_gain * input.actual_vy),
                    -adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit),
                    adaptVelocityFeedbackCommand(config, vision_servo_velocity_control, brake_limit));
                raw_command_x = command_x;
                raw_command_y = command_y;
                if (config.control_axis == "x")
                {
                    command_y = 0.0;
                }
                else if (config.control_axis == "y")
                {
                    command_x = 0.0;
                }
            }
            else
            {
                if (!input.velocity_fresh && config.hold_brake_enable)
                {
                    servo_phase = "hold_no_velocity";
                    state.vision_center_lock_active = false;
                    state.vision_center_lock_speed_over_count = 0;
                    state.vision_stable_since = ros::Time(0);
                    state.vision_stable_logged = false;
                }
                else
                {
                    if (state.vision_stable_since.isZero())
                    {
                        state.vision_stable_since = input.now;
                    }
                    output.stable_elapsed = state.vision_stable_since.isZero()
                        ? 0.0
                        : (input.now - state.vision_stable_since).toSec();
                    if (center_lock_enabled &&
                        input.velocity_fresh &&
                        output.stable_elapsed >= center_lock_time)
                    {
                        state.vision_center_lock_active = true;
                        state.vision_center_lock_speed_over_count = 0;
                        servo_phase = "center_locked";
                    }
                    else
                    {
                        servo_phase = center_lock_enabled && input.velocity_fresh
                            ? "capture_hold"
                            : "stable_hold";
                    }
                    if (!state.vision_stable_logged &&
                        output.stable_elapsed >= std::max(0.0, config.stable_time))
                    {
                        ROS_INFO(
                            "vision_servo: stable target hold %.2fs err=(%.1f, %.1f) speed=%.3f m/s",
                            output.stable_elapsed,
                            error_x_px,
                            error_y_px,
                            input.actual_speed_xy);
                        state.vision_stable_logged = true;
                    }
                }
            }
        }
    }

    if (vision_servo_velocity_control)
    {
        const double anchor_delta_x =
            input.pose_for_log.pose.position.x - state.hover_anchor_pose.pose.position.x;
        const double anchor_delta_y =
            input.pose_for_log.pose.position.y - state.hover_anchor_pose.pose.position.y;
        const double max_offset = std::max(0.0, config.max_total_offset_xy);
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

        state.velocity_servo_target = makeVelocityTargetLocal(state.hover_anchor_pose, command_x, command_y);
        state.velocity_servo_active = true;
        state.servo_offset_x = anchor_delta_x;
        state.servo_offset_y = anchor_delta_y;
    }
    else if (outside_deadband ||
             std::abs(command_x) > 1e-9 ||
             std::abs(command_y) > 1e-9)
    {
        const double next_servo_offset_x = clampValue(
            state.servo_offset_x + command_x,
            -config.max_total_offset_xy,
            config.max_total_offset_xy);
        const double next_servo_offset_y = clampValue(
            state.servo_offset_y + command_y,
            -config.max_total_offset_xy,
            config.max_total_offset_xy);
        command_x = next_servo_offset_x - state.servo_offset_x;
        command_y = next_servo_offset_y - state.servo_offset_y;
        state.servo_offset_x = next_servo_offset_x;
        state.servo_offset_y = next_servo_offset_y;

        state.velocity_servo_active = false;
        state.velocity_servo_target = makeHoldPositionTargetLocal(target_pose);
        target_pose.pose.position.x = state.hover_anchor_pose.pose.position.x + state.servo_offset_x;
        target_pose.pose.position.y = state.hover_anchor_pose.pose.position.y + state.servo_offset_y;
        target_pose.pose.position.z = state.hover_anchor_pose.pose.position.z;
        target_pose.pose.orientation = state.hover_anchor_pose.pose.orientation;
    }
    else
    {
        state.velocity_servo_active = false;
        state.velocity_servo_target = makeHoldPositionTargetLocal(target_pose);
    }

    output.servo_phase = servo_phase;
    output.servo_filtered_error_x_px = state.servo_filtered_error_x_px;
    output.servo_filtered_error_y_px = state.servo_filtered_error_y_px;
    output.derivative_error_x_px = derivative_error_x_px;
    output.derivative_error_y_px = derivative_error_y_px;
    output.damped_error_x_px = damped_error_x_px;
    output.damped_error_y_px = damped_error_y_px;
    output.raw_command_x = raw_command_x;
    output.raw_command_y = raw_command_y;
    output.damping_command_x = damping_command_x;
    output.damping_command_y = damping_command_y;
    output.outer_slowdown_log_x = outer_slowdown_log_x;
    output.outer_slowdown_log_y = outer_slowdown_log_y;
    output.command_x = command_x;
    output.command_y = command_y;
    output.error_jump_px = error_jump_px;
    output.error_jump_rejected = error_jump_rejected;
    output.stable_elapsed = std::max(output.stable_elapsed, 0.0);

    state.last_servo_update_time = input.now;
    state.servo_previous_command_x = command_x;
    state.servo_previous_command_y = command_y;
}

}  // namespace electronic_fly
