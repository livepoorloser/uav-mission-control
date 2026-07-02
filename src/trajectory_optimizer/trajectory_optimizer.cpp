// 功能：轨迹优化模块，生成满足速度、加速度和 yaw 约束的平滑轨迹。

#include "electronic_fly/trajectory_optimizer/trajectory_optimizer.h"

#include <algorithm>
#include <cmath>

namespace electronic_fly
{
namespace
{

double normalizeAngle(double angle)
{
    const double pi = std::acos(-1.0);
    while (angle > pi)
    {
        angle -= 2.0 * pi;
    }
    while (angle < -pi)
    {
        angle += 2.0 * pi;
    }
    return angle;
}

double clampValue(double value, double min_value, double max_value)
{
    return std::max(min_value, std::min(value, max_value));
}

double distance3d(const Waypoint& from, const Waypoint& to)
{
    const double dx = to.pos[0] - from.pos[0];
    const double dy = to.pos[1] - from.pos[1];
    const double dz = to.pos[2] - from.pos[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool hasValidPosition(const Waypoint& waypoint)
{
    return waypoint.pos.size() == 3;
}

double minimumJerkBlend(double tau)
{
    const double tau2 = tau * tau;
    const double tau3 = tau2 * tau;
    const double tau4 = tau3 * tau;
    const double tau5 = tau4 * tau;
    return 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5;
}

QuinticTrajectory::QuinticPolynomial makeMinimumJerkPolynomial(double start, double goal, double duration)
{
    QuinticTrajectory::QuinticPolynomial polynomial;
    if (duration <= 1e-6)
    {
        polynomial.a0 = goal;
        return polynomial;
    }

    const double delta = goal - start;
    const double duration_sq = duration * duration;
    const double duration_cu = duration_sq * duration;
    const double duration_qu = duration_cu * duration;
    const double duration_5 = duration_qu * duration;

    polynomial.a0 = start;
    polynomial.a1 = 0.0;
    polynomial.a2 = 0.0;
    polynomial.a3 = 10.0 * delta / duration_cu;
    polynomial.a4 = -15.0 * delta / duration_qu;
    polynomial.a5 = 6.0 * delta / duration_5;
    return polynomial;
}

double computeDuration(
    const Waypoint& start,
    const Waypoint& goal,
    const TrajectoryOptimizerConfig& config)
{
    const double distance = distance3d(start, goal);
    const double yaw_delta = std::abs(normalizeAngle(goal.yaw - start.yaw));
    const double nominal_speed = std::max(config.nominal_speed, 1e-3);
    const double max_acceleration = std::max(config.max_acceleration, 1e-3);
    const double max_yaw_rate = std::max(config.max_yaw_rate, 1e-3);

    const double duration_from_speed = distance / nominal_speed;
    const double duration_from_acceleration = 2.0 * std::sqrt(distance / max_acceleration);
    const double duration_from_yaw = config.interpolate_yaw ? yaw_delta / max_yaw_rate : 0.0;

    double duration = std::max(config.min_segment_time, duration_from_speed);
    duration = std::max(duration, duration_from_acceleration);
    duration = std::max(duration, duration_from_yaw);

    if (config.max_segment_time > 0.0)
    {
        duration = std::min(duration, config.max_segment_time);
    }

    return std::max(duration, 0.2);
}

}  // namespace

bool QuinticTrajectory::valid() const
{
    return valid_;
}

bool QuinticTrajectory::isFinished(double time_from_start) const
{
    return !valid_ || time_from_start >= duration_;
}

double QuinticTrajectory::duration() const
{
    return duration_;
}

const Waypoint& QuinticTrajectory::goal() const
{
    return goal_;
}

double QuinticTrajectory::QuinticPolynomial::evaluate(double t) const
{
    return (((a5 * t + a4) * t + a3) * t + a2) * t * t + a1 * t + a0;
}

TrajectorySample QuinticTrajectory::sample(double time_from_start) const
{
    TrajectorySample sample_point;
    sample_point.pos.resize(3, 0.0);

    if (!valid_)
    {
        return sample_point;
    }

    const double clamped_time = clampValue(time_from_start, 0.0, duration_);
    const double tau = duration_ > 1e-6 ? clamped_time / duration_ : 1.0;
    const double blend = minimumJerkBlend(clampValue(tau, 0.0, 1.0));

    sample_point.pos[0] = start_.pos[0] + (goal_.pos[0] - start_.pos[0]) * blend;
    sample_point.pos[1] = start_.pos[1] + (goal_.pos[1] - start_.pos[1]) * blend;
    sample_point.pos[2] = start_.pos[2] + (goal_.pos[2] - start_.pos[2]) * blend;

    if (interpolate_yaw_)
    {
        sample_point.yaw = start_.yaw + normalizeAngle(goal_.yaw - start_.yaw) * blend;
    }
    else
    {
        sample_point.yaw = goal_.yaw;
    }

    return sample_point;
}

QuinticTrajectory buildTrajectory(
    const Waypoint& start,
    const Waypoint& goal,
    const TrajectoryOptimizerConfig& config)
{
    QuinticTrajectory trajectory;
    trajectory.interpolate_yaw_ = config.interpolate_yaw;
    trajectory.start_ = start;
    trajectory.goal_ = goal;

    if (!hasValidPosition(start) || !hasValidPosition(goal))
    {
        return trajectory;
    }

    trajectory.duration_ = computeDuration(start, goal, config);
    trajectory.x_ = makeMinimumJerkPolynomial(start.pos[0], goal.pos[0], trajectory.duration_);
    trajectory.y_ = makeMinimumJerkPolynomial(start.pos[1], goal.pos[1], trajectory.duration_);
    trajectory.z_ = makeMinimumJerkPolynomial(start.pos[2], goal.pos[2], trajectory.duration_);

    if (config.interpolate_yaw)
    {
        const double yaw_goal = start.yaw + normalizeAngle(goal.yaw - start.yaw);
        trajectory.yaw_ = makeMinimumJerkPolynomial(start.yaw, yaw_goal, trajectory.duration_);
    }
    else
    {
        trajectory.yaw_ = makeMinimumJerkPolynomial(goal.yaw, goal.yaw, trajectory.duration_);
    }

    trajectory.valid_ = true;
    return trajectory;
}

}  // namespace electronic_fly
