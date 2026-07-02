#pragma once
#include <ros/ros.h>
#include <string>
#include <vector>

namespace electronic_config{
class DSconfig
{
public:
    struct PathPoints
    {
        std::vector<double> take_off_point;
        std::vector<double> points;
        std::vector<double> perform_high;
        std::vector<double> land_point;
        int num;
        int perform_time;
    };
    struct VehiclePose
    {
        bool useodometry;
        bool useLocal;
        bool useVision;
        std::string subscribe_odometry_topic;
        std::string subscribe_Local_topic;
        std::string subscribe_Vision_topic;
    };

    PathPoints path_point;
    VehiclePose Vehicle_Pose;
    DSconfig(){};
    void getParamters(const ros::NodeHandle &nh);
private:
    template <typename TName, typename TVal>
	void read_essential_param(const ros::NodeHandle &nh, const TName &name, TVal &val)
	{
		if (nh.getParam(name, val))
		{
		}
		else
		{
			ROS_ERROR_STREAM("Read param: " << name << " failed.");
			ROS_BREAK();
		}
	};

};


void DSconfig::getParamters(const ros::NodeHandle &nh){
    read_essential_param(nh, "PathPoints/take_off_point", path_point.take_off_point);
    read_essential_param(nh, "PathPoints/points", path_point.points);
    read_essential_param(nh, "PathPoints/perform_high", path_point.perform_high);
    read_essential_param(nh, "PathPoints/perform_time", path_point.perform_time);
    read_essential_param(nh, "PathPoints/num", path_point.num);
    read_essential_param(nh, "PathPoints/land_point", path_point.land_point);
    read_essential_param(nh, "VehiclePose/useodometry", Vehicle_Pose.useodometry);
    read_essential_param(nh, "VehiclePose/useLocal", Vehicle_Pose.useLocal);
    read_essential_param(nh, "VehiclePose/useVision", Vehicle_Pose.useVision);
    read_essential_param(nh, "VehiclePose/subscribe_odometry_topic", Vehicle_Pose.subscribe_odometry_topic);
    read_essential_param(nh, "VehiclePose/subscribe_Local_topic", Vehicle_Pose.subscribe_Local_topic);
    read_essential_param(nh, "VehiclePose/subscribe_Vision_topic", Vehicle_Pose.subscribe_Vision_topic);
}

}
