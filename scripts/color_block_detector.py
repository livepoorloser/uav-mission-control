#!/usr/bin/env python3
# 功能：基于颜色阈值的目标检测节点，用于轻量视觉检测。


import json
import logging
import threading

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


DEFAULT_CLASSES = ["tiger", "peacock", "monkey", "elephant", "wolf"]

g_pose_lock = threading.Lock()
g_current_pose = None
g_vision_enabled = True
g_gate_dirty = True


def get_detector_param(name, default):
    private_name = "~{}".format(name)
    if rospy.has_param(private_name):
        return rospy.get_param(private_name)

    global_name = "/animal_detector/{}".format(name)
    if rospy.has_param(global_name):
        return rospy.get_param(global_name)
    return default


def repair_logging_levels():
    canonical_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    alias_levels = {
        "FATAL": logging.FATAL,
        "WARN": logging.WARNING,
    }
    for name, level in canonical_levels.items():
        logging._nameToLevel[name] = level
        logging._levelToName[level] = name
    for name, level in alias_levels.items():
        logging._nameToLevel[name] = level


def pose_cb(msg):
    global g_current_pose
    with g_pose_lock:
        g_current_pose = msg


def vision_check_cb(msg):
    global g_vision_enabled, g_gate_dirty
    next_enabled = bool(msg.data)
    if next_enabled != g_vision_enabled:
        g_gate_dirty = True
    g_vision_enabled = next_enabled


def get_pose_dict():
    with g_pose_lock:
        pose = g_current_pose
    if pose is None:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "x": round(float(pose.pose.position.x), 3),
        "y": round(float(pose.pose.position.y), 3),
        "z": round(float(pose.pose.position.z), 3),
    }


def set_detection_params(class_names, counts):
    total = 0
    for index, name in enumerate(class_names, start=1):
        value = int(counts.get(name, 0))
        rospy.set_param("vision/animal_kind_{}".format(index), value)
        total += value
    rospy.set_param("vision/animal_total", total)
    rospy.set_param("vision/animal_flag", 1 if total > 0 else 0)


def build_summary(class_names, counts, diffs, tracking):
    total = sum(int(counts.get(name, 0)) for name in class_names)
    return {
        "timestamp": round(rospy.Time.now().to_sec(), 3),
        "source": "opencv_color",
        "counts": {name: int(counts.get(name, 0)) for name in class_names},
        "total": total,
        "pose": get_pose_dict(),
        "center_diffs": diffs,
        "tracking": tracking,
    }


def publish_summary(summary, class_names, detection_pub, control_pub):
    counts = summary.get("counts", {})
    set_detection_params(class_names, counts)
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    rospy.set_param("vision/class_names", class_names)
    rospy.set_param("vision/last_detection_json", summary_json)
    detection_pub.publish(String(data=summary_json))
    control_pub.publish(Bool(data=False))
    return summary_json


def empty_tracking():
    return {
        "target_count": 0,
        "avg_dx": 0.0,
        "avg_dy": 0.0,
        "primary_dx": 0.0,
        "primary_dy": 0.0,
        "primary_label": "",
        "primary_score": 0.0,
        "primary_area_px": 0.0,
        "primary_u": 0.0,
        "primary_v": 0.0,
        "primary_raw_u": 0.0,
        "primary_raw_v": 0.0,
        "aim_px_x": 0,
        "aim_px_y": 0,
        "valid": False,
    }


def publish_empty_summary(class_names, detection_pub, control_pub):
    counts = {name: 0 for name in class_names}
    return publish_summary(
        build_summary(class_names, counts, [], empty_tracking()),
        class_names,
        detection_pub,
        control_pub)


def build_center_region(frame_width, frame_height, width_ratio, height_ratio):
    x1 = int(frame_width * (1.0 - width_ratio) / 2.0)
    y1 = int(frame_height * (1.0 - height_ratio) / 2.0)
    x2 = frame_width - x1
    y2 = frame_height - y1
    return x1, y1, x2, y2


def center_contains(point, center_region):
    x, y = point
    return center_region[0] <= x <= center_region[2] and center_region[1] <= y <= center_region[3]


def make_hsv_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max, blur_kernel):
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_min = int(np.clip(s_min, 0, 255))
    s_max = int(np.clip(s_max, 0, 255))
    v_min = int(np.clip(v_min, 0, 255))
    v_max = int(np.clip(v_max, 0, 255))
    h_min = int(np.clip(h_min, 0, 179))
    h_max = int(np.clip(h_max, 0, 179))

    if h_min <= h_max:
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    lower_a = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_a = np.array([179, s_max, v_max], dtype=np.uint8)
    lower_b = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_b = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower_a, upper_a), cv2.inRange(hsv, lower_b, upper_b))


def clean_mask(mask, open_kernel, close_kernel):
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_color_detections(frame, mask, label, min_area, max_area, max_targets, aim_y_ratio,
                          use_center_region_filter, center_region):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    detections = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or (max_area > 0.0 and area > max_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aim_x = float(x + 0.5 * w)
        aim_y = float(y + np.clip(aim_y_ratio, 0.0, 1.0) * h)
        if use_center_region_filter and not center_contains((aim_x, aim_y), center_region):
            continue

        detections.append({
            "box": (x, y, x + w, y + h),
            "label": label,
            "score": 1.0,
            "area_px": area,
            "u": aim_x,
            "v": aim_y,
            "dx": float(aim_x - image_center[0]),
            "dy": float(aim_y - image_center[1]),
        })

    detections.sort(key=lambda item: item["area_px"], reverse=True)
    return detections[:max(1, int(max_targets))]


def build_detection_payload(class_names, detections):
    counts = {name: 0 for name in class_names}
    diffs = []
    sum_dx = 0.0
    sum_dy = 0.0
    primary = None

    for detection in detections:
        label = detection["label"]
        counts[label] = int(counts.get(label, 0)) + 1
        sum_dx += detection["dx"]
        sum_dy += detection["dy"]
        diffs.append({
            "class": label,
            "dx": round(float(detection["dx"]), 1),
            "dy": round(float(detection["dy"]), 1),
            "score": round(float(detection["score"]), 4),
            "area_px": round(float(detection["area_px"]), 1),
            "u": round(float(detection["u"]), 3),
            "v": round(float(detection["v"]), 3),
            "raw_u": round(float(detection["u"]), 3),
            "raw_v": round(float(detection["v"]), 3),
            "undistorted": False,
        })
        if primary is None or detection["area_px"] > primary["area_px"]:
            primary = detection

    if not detections or primary is None:
        return counts, [], empty_tracking()

    tracking = {
        "target_count": len(detections),
        "avg_dx": round(sum_dx / float(len(detections)), 3),
        "avg_dy": round(sum_dy / float(len(detections)), 3),
        "primary_dx": round(float(primary["dx"]), 3),
        "primary_dy": round(float(primary["dy"]), 3),
        "primary_label": primary["label"],
        "primary_score": round(float(primary["score"]), 4),
        "primary_area_px": round(float(primary["area_px"]), 1),
        "primary_u": round(float(primary["u"]), 3),
        "primary_v": round(float(primary["v"]), 3),
        "primary_raw_u": round(float(primary["u"]), 3),
        "primary_raw_v": round(float(primary["v"]), 3),
        "aim_px_x": int(round(float(primary["u"]))),
        "aim_px_y": int(round(float(primary["v"]))),
        "valid": True,
    }
    return counts, diffs, tracking


def draw_overlay(frame, detections, center_region, use_center_region_filter, paused, summary_text):
    output = frame.copy()
    image_center = (output.shape[1] // 2, output.shape[0] // 2)
    cv2.drawMarker(output, image_center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    if use_center_region_filter:
        x1, y1, x2, y2 = center_region
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0.0, output)

    if paused:
        cv2.putText(output, "DETECTION PAUSED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if summary_text:
        cv2.putText(output, summary_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        u = int(round(detection["u"]))
        v = int(round(detection["v"]))
        label = "{} {:.0f}px dx={:.0f} dy={:.0f}".format(
            detection["label"], detection["area_px"], detection["dx"], detection["dy"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(output, (u, v), 4, (0, 255, 0), -1)
        cv2.putText(output, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return output


def cv2_to_image_msg(frame, frame_id):
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = frame.strides[0]
    msg.data = frame.tobytes()
    return msg


def normalize_camera_source(camera_source):
    if isinstance(camera_source, str):
        stripped = camera_source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return int(camera_source)


def main():
    global g_gate_dirty, g_vision_enabled

    repair_logging_levels()
    rospy.init_node("color_block_detector")

    class_names = get_detector_param("classes", DEFAULT_CLASSES)
    if not class_names:
        class_names = list(DEFAULT_CLASSES)

    label = str(get_detector_param("label", get_detector_param("color_label", "tiger")))
    if label not in class_names:
        class_names = list(class_names) + [label]

    g_vision_enabled = bool(get_detector_param("enabled_by_default", True))
    camera_source = normalize_camera_source(get_detector_param("camera_id", 0))
    frame_width = int(get_detector_param("frame_width", 640))
    frame_height = int(get_detector_param("frame_height", 480))
    loop_hz = float(get_detector_param("loop_hz", 15.0))
    pose_topic = get_detector_param("pose_topic", "mavros/local_position/pose")

    h_min = int(get_detector_param("h_min", 170))
    h_max = int(get_detector_param("h_max", 10))
    s_min = int(get_detector_param("s_min", 80))
    s_max = int(get_detector_param("s_max", 255))
    v_min = int(get_detector_param("v_min", 80))
    v_max = int(get_detector_param("v_max", 255))
    blur_kernel = int(get_detector_param("blur_kernel", 3))
    open_kernel = int(get_detector_param("open_kernel", 3))
    close_kernel = int(get_detector_param("close_kernel", 5))
    min_area = float(get_detector_param("min_area", 500.0))
    max_area = float(get_detector_param("max_area", 25000.0))
    max_targets = int(get_detector_param("max_targets", 1))
    aim_y_ratio = float(get_detector_param("aim_y_ratio", 0.5))
    use_center_region_filter = bool(get_detector_param("use_center_region_filter", False))
    region_width_ratio = float(get_detector_param("region_width_ratio", 1.0))
    region_height_ratio = float(get_detector_param("region_height_ratio", 1.0))
    verbose_log = bool(get_detector_param("verbose_log", True))
    log_period = float(get_detector_param("log_period", 0.5))

    show_window = bool(get_detector_param("show_window", False))
    publish_raw_image = bool(get_detector_param("publish_raw_image", True))
    raw_image_topic = get_detector_param("raw_image_topic", "vision/raw_image")
    raw_frame_id = get_detector_param("raw_frame_id", "color_block_camera")
    publish_debug_image = bool(get_detector_param("publish_debug_image", True))
    debug_image_topic = get_detector_param("debug_image_topic", "vision/detection_image")
    debug_frame_id = get_detector_param("debug_frame_id", "color_block_camera")

    rospy.Subscriber(pose_topic, PoseStamped, pose_cb, queue_size=10)
    rospy.Subscriber("vision/check", Bool, vision_check_cb, queue_size=10)
    detection_pub = rospy.Publisher("vision/detections_json", String, queue_size=10)
    control_pub = rospy.Publisher("vision/control", Bool, queue_size=10)
    raw_image_pub = rospy.Publisher(raw_image_topic, Image, queue_size=1) if publish_raw_image else None
    debug_image_pub = rospy.Publisher(debug_image_topic, Image, queue_size=1) if publish_debug_image else None

    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        rospy.logerr("color_block_detector: failed to open camera %s", str(camera_source))
        return
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    publish_empty_summary(class_names, detection_pub, control_pub)
    rospy.loginfo(
        "color_block_detector: ready label=%s hsv=(%d..%d,%d..%d,%d..%d) camera=%s",
        label,
        h_min,
        h_max,
        s_min,
        s_max,
        v_min,
        v_max,
        str(camera_source),
    )

    last_detections = []
    last_summary_text = ""
    rate = rospy.Rate(loop_hz)

    try:
        while not rospy.is_shutdown():
            if g_gate_dirty:
                last_detections = []
                last_summary_text = ""
                publish_empty_summary(class_names, detection_pub, control_pub)
                g_gate_dirty = False

            ok, frame = capture.read()
            if not ok or frame is None:
                rospy.logwarn_throttle(2.0, "color_block_detector: failed to read camera frame")
                rate.sleep()
                continue

            if publish_raw_image:
                raw_image_pub.publish(cv2_to_image_msg(frame, raw_frame_id))

            center_region = build_center_region(
                frame.shape[1], frame.shape[0], region_width_ratio, region_height_ratio)

            detections = []
            if g_vision_enabled:
                mask = make_hsv_mask(frame, h_min, h_max, s_min, s_max, v_min, v_max, blur_kernel)
                mask = clean_mask(mask, open_kernel, close_kernel)
                detections = find_color_detections(
                    frame,
                    mask,
                    label,
                    min_area,
                    max_area,
                    max_targets,
                    aim_y_ratio,
                    use_center_region_filter,
                    center_region)

                counts, diffs, tracking = build_detection_payload(class_names, detections)
                publish_summary(
                    build_summary(class_names, counts, diffs, tracking),
                    class_names,
                    detection_pub,
                    control_pub)
                if verbose_log:
                    if tracking.get("valid", False):
                        primary = max(detections, key=lambda item: item["area_px"])
                        x1, y1, x2, y2 = primary["box"]
                        rospy.loginfo_throttle(
                            log_period,
                            "color_block: targets=%d primary=%s u=%.1f v=%.1f dx=%.1f dy=%.1f area=%.1f box=(%d,%d,%d,%d) hsv=(%d..%d,%d..%d,%d..%d)",
                            len(detections),
                            primary["label"],
                            primary["u"],
                            primary["v"],
                            primary["dx"],
                            primary["dy"],
                            primary["area_px"],
                            x1,
                            y1,
                            x2,
                            y2,
                            h_min,
                            h_max,
                            s_min,
                            s_max,
                            v_min,
                            v_max,
                        )
                    else:
                        rospy.loginfo_throttle(
                            log_period,
                            "color_block: targets=0 hsv=(%d..%d,%d..%d,%d..%d) min_area=%.1f max_area=%.1f",
                            h_min,
                            h_max,
                            s_min,
                            s_max,
                            v_min,
                            v_max,
                            min_area,
                            max_area,
                        )
                last_detections = detections
                last_summary_text = " ".join(
                    "{}:{}".format(name, counts.get(name, 0)) for name in class_names if counts.get(name, 0) > 0
                )
            else:
                control_pub.publish(Bool(data=False))

            if show_window or publish_debug_image:
                output_detections = last_detections if g_vision_enabled else []
                output = draw_overlay(
                    frame,
                    output_detections,
                    center_region,
                    use_center_region_filter,
                    not g_vision_enabled,
                    last_summary_text)
                if publish_debug_image:
                    debug_image_pub.publish(cv2_to_image_msg(output, debug_frame_id))
                if show_window:
                    cv2.imshow("OpenCV Color Block Detection", output)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break

            rate.sleep()
    finally:
        capture.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
