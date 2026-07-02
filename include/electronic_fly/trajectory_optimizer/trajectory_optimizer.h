#pragma once

#include <string>
#include <vector>

namespace electronic_fly
{

struct Waypoint
{
    std::vector<double> pos;
    double yaw = 0.0;
    double hover_time = 0.0;
    bool inspect = true;
    int grid_row = -1;
    int grid_col = -1;
    std::string grid_code;
};

struct TrajectoryOptimizerConfig
{
    bool enable = false;
    bool interpolate_yaw = true;
    double nominal_speed = 0.8;
    double max_acceleration = 0.8;
    double min_segment_time = 1.0;
    double max_segment_time = 0.0;
    double max_yaw_rate = 0.8;
    double max_tracking_error = 0.60;
};

struct TrajectorySample
{
    std::vector<double> pos;
    double yaw = 0.0;
};

class QuinticTrajectory
{
public:
    struct QuinticPolynomial
    {
        double a0 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
        double a3 = 0.0;
        double a4 = 0.0;
        double a5 = 0.0;

        double evaluate(double t) const;
    };

    QuinticTrajectory() = default;

    bool valid() const;
    bool isFinished(double time_from_start) const;
    double duration() const;
    const Waypoint& goal() const;
    TrajectorySample sample(double time_from_start) const;

private:
    bool valid_ = false;
    bool interpolate_yaw_ = true;
    double duration_ = 0.0;
    Waypoint start_;
    Waypoint goal_;
    QuinticPolynomial x_;
    QuinticPolynomial y_;
    QuinticPolynomial z_;
    QuinticPolynomial yaw_;

    friend QuinticTrajectory buildTrajectory(
        const Waypoint& start,
        const Waypoint& goal,
        const TrajectoryOptimizerConfig& config);
};

QuinticTrajectory buildTrajectory(
    const Waypoint& start,
    const Waypoint& goal,
    const TrajectoryOptimizerConfig& config);

}  // namespace electronic_fly
