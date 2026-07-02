// 功能：基于 ONNX/OpenCV 的动物检测节点，发布视觉检测结果。

#include <ros/ros.h>
#include <ros/package.h>

#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/PositionTarget.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "electronic_fly/json_utils.h"

namespace
{

struct Detection
{
    cv::Rect box;
    std::string label;
    float score = 0.0f;
};

geometry_msgs::PoseStamped g_current_pose;
bool g_have_pose = false;
bool g_vision_enabled = false;
bool g_vision_gate_dirty = true;

void poseCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    g_current_pose = *msg;
    g_have_pose = true;
}

void visionCheckCb(const std_msgs::Bool::ConstPtr& msg)
{
    const bool next_enabled = msg->data;
    if (next_enabled != g_vision_enabled)
    {
        g_vision_gate_dirty = true;
    }
    g_vision_enabled = next_enabled;
}

class YoloV5OnnxDetector
{
public:
    YoloV5OnnxDetector(
        const std::string& model_path,
        std::vector<std::string> class_names,
        int input_size,
        float conf_threshold,
        float iou_threshold)
        : class_names_(std::move(class_names)),
          input_size_(input_size),
          conf_threshold_(conf_threshold),
          iou_threshold_(iou_threshold),
          net_(cv::dnn::readNetFromONNX(model_path))
    {
    }

    std::vector<Detection> infer(const cv::Mat& frame) const
    {
        float scale = 1.0f;
        int pad_left = 0;
        int pad_top = 0;
        const cv::Mat letterboxed = letterbox(frame, scale, pad_left, pad_top);
        cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat output = net_.forward();

        if (output.dims == 3)
        {
            output = output.reshape(1, output.size[1]);
        }
        else if (output.dims > 3)
        {
            output = output.reshape(1, output.total() / output.size[output.dims - 1]);
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int row = 0; row < output.rows; ++row)
        {
            const float* prediction = output.ptr<float>(row);
            if (output.cols < 6)
            {
                continue;
            }

            const float objectness = prediction[4];
            int best_class = -1;
            float best_score = 0.0f;
            for (int col = 5; col < output.cols; ++col)
            {
                if (prediction[col] > best_score)
                {
                    best_score = prediction[col];
                    best_class = col - 5;
                }
            }

            const float confidence = objectness * best_score;
            if (confidence < conf_threshold_ || best_class < 0 || best_class >= static_cast<int>(class_names_.size()))
            {
                continue;
            }

            const float cx = prediction[0];
            const float cy = prediction[1];
            const float w = prediction[2];
            const float h = prediction[3];

            int x1 = static_cast<int>(std::round((cx - w / 2.0f - pad_left) / scale));
            int y1 = static_cast<int>(std::round((cy - h / 2.0f - pad_top) / scale));
            int x2 = static_cast<int>(std::round((cx + w / 2.0f - pad_left) / scale));
            int y2 = static_cast<int>(std::round((cy + h / 2.0f - pad_top) / scale));

            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));
            if (x2 <= x1 || y2 <= y1)
            {
                continue;
            }

            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.push_back(confidence);
            class_ids.push_back(best_class);
        }

        std::vector<Detection> detections;
        if (boxes.empty())
        {
            return detections;
        }

        std::vector<int> keep_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, iou_threshold_, keep_indices);
        for (int index : keep_indices)
        {
            Detection detection;
            detection.box = boxes[index];
            detection.label = class_names_[class_ids[index]];
            detection.score = scores[index];
            detections.push_back(detection);
        }
        return detections;
    }

private:
    cv::Mat letterbox(const cv::Mat& frame, float& scale, int& pad_left, int& pad_top) const
    {
        const int input_w = frame.cols;
        const int input_h = frame.rows;
        scale = std::min(
            static_cast<float>(input_size_) / static_cast<float>(input_h),
            static_cast<float>(input_size_) / static_cast<float>(input_w));

        const int new_w = static_cast<int>(std::round(input_w * scale));
        const int new_h = static_cast<int>(std::round(input_h * scale));
        const int pad_w = input_size_ - new_w;
        const int pad_h = input_size_ - new_h;
        pad_left = pad_w / 2;
        const int pad_right = pad_w - pad_left;
        pad_top = pad_h / 2;
        const int pad_bottom = pad_h - pad_top;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat bordered;
        cv::copyMakeBorder(
            resized, bordered, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return bordered;
    }

    std::vector<std::string> class_names_;
    int input_size_ = 640;
    float conf_threshold_ = 0.25f;
    float iou_threshold_ = 0.45f;
    mutable cv::dnn::Net net_;
};

std::string getDefaultModelPath()
{
    const std::string package_path = ros::package::getPath("drone_code");
    const std::vector<std::string> candidates = {
        package_path + "/yolov5-master/yolov5s(2).onnx",
        package_path + "/yolov5-master/runs/train/exp13/weights/best.onnx",
    };

    for (const auto& candidate : candidates)
    {
        std::ifstream stream(candidate.c_str());
        if (stream.good())
        {
            return candidate;
        }
    }
    return candidates.front();
}

std::vector<std::string> getDefaultClassNames()
{
    return {"tiger", "peacock", "monkey", "elephant", "wolf"};
}

cv::Rect buildCenterRegion(int frame_width, int frame_height, double width_ratio, double height_ratio)
{
    const int x1 = static_cast<int>(frame_width * (1.0 - width_ratio) / 2.0);
    const int y1 = static_cast<int>(frame_height * (1.0 - height_ratio) / 2.0);
    const int x2 = frame_width - x1;
    const int y2 = frame_height - y1;
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

bool isInsideCenterRegion(const Detection& detection, const cv::Rect& center_region)
{
    const cv::Point center = (detection.box.tl() + detection.box.br()) / 2;
    return center_region.contains(center);
}

std::vector<Detection> filterByArea(const std::vector<Detection>& detections, double min_area, double max_area)
{
    std::vector<Detection> filtered;
    for (const auto& detection : detections)
    {
        const double area = static_cast<double>(detection.box.area());
        if (area >= min_area && area <= max_area)
        {
            filtered.push_back(detection);
        }
    }
    return filtered;
}

void setDetectionParams(const std::vector<std::string>& class_names, const std::map<std::string, int>& counts, ros::NodeHandle& nh)
{
    int total = 0;
    for (std::size_t index = 0; index < class_names.size(); ++index)
    {
        const int value = counts.count(class_names[index]) ? counts.at(class_names[index]) : 0;
        nh.setParam("vision/animal_kind_" + std::to_string(index + 1), value);
        total += value;
    }
    nh.setParam("vision/animal_total", total);
    nh.setParam("vision/animal_flag", total > 0 ? 1 : 0);
}

boost::property_tree::ptree buildDetectionSummary(
    const std::vector<std::string>& class_names,
    const std::map<std::string, int>& counts,
    const boost::property_tree::ptree& diffs)
{
    boost::property_tree::ptree summary;
    summary.put("timestamp", ros::Time::now().toSec());

    boost::property_tree::ptree counts_tree;
    int total = 0;
    for (const auto& name : class_names)
    {
        const int value = counts.count(name) ? counts.at(name) : 0;
        counts_tree.put(name, value);
        total += value;
    }

    summary.add_child("counts", counts_tree);
    summary.put("total", total);

    boost::property_tree::ptree pose_tree;
    pose_tree.put("x", g_current_pose.pose.position.x);
    pose_tree.put("y", g_current_pose.pose.position.y);
    pose_tree.put("z", g_current_pose.pose.position.z);
    summary.add_child("pose", pose_tree);
    summary.add_child("center_diffs", diffs);
    return summary;
}

std::string publishSummary(
    const boost::property_tree::ptree& summary,
    const std::vector<std::string>& class_names,
    ros::NodeHandle& nh,
    ros::Publisher& detection_pub)
{
    std::map<std::string, int> counts;
    for (const auto& name : class_names)
    {
        counts[name] = summary.get<int>("counts." + name, 0);
    }
    setDetectionParams(class_names, counts, nh);

    const std::string summary_json = electronic_fly::toJson(summary, false);
    std_msgs::String message;
    message.data = summary_json;
    detection_pub.publish(message);
    nh.setParam("vision/last_detection_json", summary_json);
    return summary_json;
}

std::string clearDetectionState(
    const std::vector<std::string>& class_names,
    ros::NodeHandle& nh,
    ros::Publisher& detection_pub)
{
    boost::property_tree::ptree empty_diffs;
    std::map<std::string, int> empty_counts;
    for (const auto& name : class_names)
    {
        empty_counts[name] = 0;
    }
    const boost::property_tree::ptree summary = buildDetectionSummary(class_names, empty_counts, empty_diffs);
    return publishSummary(summary, class_names, nh, detection_pub);
}

cv::Mat drawOverlay(
    cv::Mat frame,
    const std::vector<Detection>& detections,
    const cv::Rect& center_region,
    const cv::Point& image_center,
    bool paused,
    const std::string& summary_text)
{
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, center_region, cv::Scalar(255, 0, 0), 2);
    cv::addWeighted(overlay, 0.3, frame, 0.7, 0.0, frame);
    cv::drawMarker(frame, image_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 14, 2);

    if (paused)
    {
        cv::putText(frame, "DETECTION PAUSED", cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);
    }
    if (!summary_text.empty())
    {
        cv::putText(frame, summary_text, cv::Point(10, 56), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 0), 2);
    }

    for (const auto& detection : detections)
    {
        const cv::Point center = (detection.box.tl() + detection.box.br()) / 2;
        const int dx = center.x - image_center.x;
        const int dy = center.y - image_center.y;
        cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, center, 4, cv::Scalar(0, 255, 0), -1);
        const std::string label = detection.label + " " + cv::format("%.2f dx=%d dy=%d", detection.score, dx, dy);
        cv::putText(
            frame, label, cv::Point(detection.box.x, std::max(18, detection.box.y - 8)),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 2);
    }
    return frame;
}

sensor_msgs::Image matToImageMsg(const cv::Mat& frame, const std::string& frame_id)
{
    sensor_msgs::Image message;
    message.header.stamp = ros::Time::now();
    message.header.frame_id = frame_id;
    message.height = static_cast<uint32_t>(frame.rows);
    message.width = static_cast<uint32_t>(frame.cols);
    message.encoding = "bgr8";
    message.is_bigendian = false;
    message.step = static_cast<sensor_msgs::Image::_step_type>(frame.step);
    const std::size_t data_size = frame.total() * frame.elemSize();
    message.data.assign(frame.data, frame.data + data_size);
    return message;
}

void maybePublishNudge(
    const std::vector<Detection>& detections,
    const cv::Point& image_center,
    bool control_enabled,
    double control_scale,
    double control_hold_seconds,
    ros::Publisher& setpoint_pub,
    ros::Publisher& control_pub)
{
    std_msgs::Bool control_msg;
    control_msg.data = !detections.empty() && control_enabled;
    control_pub.publish(control_msg);
    if (!control_enabled || detections.empty() || !g_have_pose)
    {
        return;
    }

    double avg_dx = 0.0;
    double avg_dy = 0.0;
    for (const auto& detection : detections)
    {
        const cv::Point center = (detection.box.tl() + detection.box.br()) / 2;
        avg_dx += static_cast<double>(center.x - image_center.x);
        avg_dy += static_cast<double>(center.y - image_center.y);
    }
    avg_dx /= static_cast<double>(detections.size());
    avg_dy /= static_cast<double>(detections.size());

    mavros_msgs::PositionTarget target;
    target.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    target.type_mask =
        mavros_msgs::PositionTarget::IGNORE_VX |
        mavros_msgs::PositionTarget::IGNORE_VY |
        mavros_msgs::PositionTarget::IGNORE_VZ |
        mavros_msgs::PositionTarget::IGNORE_AFX |
        mavros_msgs::PositionTarget::IGNORE_AFY |
        mavros_msgs::PositionTarget::IGNORE_AFZ |
        mavros_msgs::PositionTarget::IGNORE_YAW |
        mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    target.position.x = g_current_pose.pose.position.x + avg_dx * control_scale;
    target.position.y = g_current_pose.pose.position.y + avg_dy * control_scale;
    target.position.z = g_current_pose.pose.position.z;

    const ros::Time start = ros::Time::now();
    ros::Rate rate(20.0);
    while (ros::ok() && (ros::Time::now() - start).toSec() < control_hold_seconds)
    {
        setpoint_pub.publish(target);
        rate.sleep();
    }
}

}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "animal_detector");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string model_path = pnh.param("model_path", std::string());
    if (model_path.empty())
    {
        model_path = getDefaultModelPath();
    }
    const int input_size = pnh.param("img_size", 640);
    const float conf_threshold = static_cast<float>(pnh.param("obj_thresh", 0.45));
    const float iou_threshold = static_cast<float>(pnh.param("nms_thresh", 0.35));
    const int camera_id = pnh.param("camera_id", 0);
    const int frame_width = pnh.param("frame_width", 640);
    const int frame_height = pnh.param("frame_height", 480);
    const double min_area = pnh.param("min_area", 500.0);
    const double max_area = pnh.param("max_area", 25000.0);
    const double region_width_ratio = pnh.param("region_width_ratio", 0.45);
    const double region_height_ratio = pnh.param("region_height_ratio", 0.60);
    const double pause_base_seconds = pnh.param("pause_base_seconds", 2.0);
    const double pause_per_object_seconds = pnh.param("pause_per_object_seconds", 1.0);
    const bool show_window = pnh.param("show_window", false);
    const bool publish_debug_image = pnh.param("publish_debug_image", true);
    const std::string debug_image_topic = pnh.param<std::string>("debug_image_topic", "vision/detection_image");
    const std::string debug_frame_id = pnh.param<std::string>("debug_frame_id", "animal_detector_camera");
    const std::string pose_topic = pnh.param<std::string>("pose_topic", "mavros/local_position/pose");
    std::vector<std::string> class_names;
    if (!pnh.getParam("classes", class_names))
    {
        class_names = getDefaultClassNames();
    }
    g_vision_enabled = pnh.param("enabled_by_default", true);
    const bool control_enabled = pnh.param("control_enabled", false);
    const double control_scale = pnh.param("control_scale", 0.001);
    const double control_hold_seconds = pnh.param("control_hold_seconds", 1.0);

    std::ifstream model_stream(model_path.c_str());
    if (!model_stream.good())
    {
        ROS_ERROR_STREAM("animal_detector: ONNX model not found: " << model_path);
        return 1;
    }

    YoloV5OnnxDetector detector(model_path, class_names, input_size, conf_threshold, iou_threshold);

    ros::Subscriber pose_sub = nh.subscribe(pose_topic, 10, poseCb);
    ros::Subscriber vision_gate_sub = nh.subscribe("vision/check", 10, visionCheckCb);
    ros::Publisher control_pub = nh.advertise<std_msgs::Bool>("vision/control", 10);
    ros::Publisher detection_pub = nh.advertise<std_msgs::String>("vision/detections_json", 10);
    ros::Publisher setpoint_pub = nh.advertise<mavros_msgs::PositionTarget>("mavros/setpoint_raw/local", 10);
    ros::Publisher debug_image_pub;
    if (publish_debug_image)
    {
        debug_image_pub = nh.advertise<sensor_msgs::Image>(debug_image_topic, 1);
    }

    cv::VideoCapture capture(camera_id);
    if (!capture.isOpened())
    {
        ROS_ERROR_STREAM("animal_detector: failed to open camera " << camera_id);
        return 1;
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);

    const cv::Point image_center(frame_width / 2, frame_height / 2);
    const cv::Rect center_region = buildCenterRegion(frame_width, frame_height, region_width_ratio, region_height_ratio);

    double paused_until = 0.0;
    std::vector<Detection> last_detections;
    std::string last_summary_text;

    nh.setParam("vision/class_names", class_names);
    clearDetectionState(class_names, nh, detection_pub);

    ROS_INFO_STREAM(
        "animal_detector: ready model=" << model_path
        << " camera_id=" << camera_id
        << " enabled=" << (g_vision_enabled ? "true" : "false"));

    ros::Rate rate(15.0);
    while (ros::ok())
    {
        ros::spinOnce();

        if (g_vision_gate_dirty)
        {
            paused_until = 0.0;
            last_detections.clear();
            last_summary_text.clear();
            clearDetectionState(class_names, nh, detection_pub);
            std_msgs::Bool control_msg;
            control_msg.data = false;
            control_pub.publish(control_msg);
            g_vision_gate_dirty = false;
        }

        cv::Mat frame;
        if (!capture.read(frame) || frame.empty())
        {
            ROS_WARN_THROTTLE(2.0, "animal_detector: failed to read camera frame");
            rate.sleep();
            continue;
        }

        const double now = ros::Time::now().toSec();
        const bool paused = now < paused_until;
        std::vector<Detection> detections;

        if (g_vision_enabled && !paused)
        {
            detections = detector.infer(frame);
            detections = filterByArea(detections, min_area, max_area);

            std::vector<Detection> centered;
            for (const auto& detection : detections)
            {
                if (isInsideCenterRegion(detection, center_region))
                {
                    centered.push_back(detection);
                }
            }
            detections.swap(centered);
            last_detections = detections;

            std::map<std::string, int> counts;
            for (const auto& name : class_names)
            {
                counts[name] = 0;
            }

            boost::property_tree::ptree diffs;
            for (const auto& detection : detections)
            {
                ++counts[detection.label];
                const cv::Point center = (detection.box.tl() + detection.box.br()) / 2;
                boost::property_tree::ptree item;
                item.put("class", detection.label);
                item.put("dx", std::round((center.x - image_center.x) * 10.0) / 10.0);
                item.put("dy", std::round((center.y - image_center.y) * 10.0) / 10.0);
                electronic_fly::appendArrayItem(diffs, item);
            }

            const boost::property_tree::ptree summary = buildDetectionSummary(class_names, counts, diffs);
            publishSummary(summary, class_names, nh, detection_pub);

            last_summary_text.clear();
            for (const auto& name : class_names)
            {
                if (counts[name] > 0)
                {
                    if (!last_summary_text.empty())
                    {
                        last_summary_text += " ";
                    }
                    last_summary_text += name + ":" + std::to_string(counts[name]);
                }
            }

            if (!detections.empty())
            {
                paused_until = now + pause_base_seconds + pause_per_object_seconds * detections.size();
                maybePublishNudge(detections, image_center, control_enabled, control_scale, control_hold_seconds, setpoint_pub, control_pub);
            }
            else
            {
                std_msgs::Bool control_msg;
                control_msg.data = false;
                control_pub.publish(control_msg);
            }
        }
        else
        {
            if (paused)
            {
                detections = last_detections;
            }
            else
            {
                std_msgs::Bool control_msg;
                control_msg.data = false;
                control_pub.publish(control_msg);
            }
        }

        if (show_window || publish_debug_image)
        {
            cv::Mat output = drawOverlay(frame, detections, center_region, image_center, paused, last_summary_text);
            if (publish_debug_image)
            {
                debug_image_pub.publish(matToImageMsg(output, debug_frame_id));
            }
            if (!show_window)
            {
                rate.sleep();
                continue;
            }
            cv::imshow("YOLOv5 ONNX Detection", output);
            if ((cv::waitKey(1) & 0xFF) == 27)
            {
                break;
            }
        }

        rate.sleep();
    }

    capture.release();
    if (show_window)
    {
        cv::destroyAllWindows();
    }
    return 0;
}
