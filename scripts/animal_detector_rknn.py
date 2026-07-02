#!/usr/bin/env python3
# 功能：基于 RKNN 的动物检测节点，发布检测图像和检测摘要。


import json
import logging
import os
import sys
import threading

import cv2
import numpy as np
import rospy
import roslib.packages
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    PACKAGE_DIR = roslib.packages.get_pkg_dir("drone_code")
except Exception:
    PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(PACKAGE_DIR)
ANIMAL_YOLO_DIR = os.path.join(WORKSPACE_DIR, "animal_yolov5")

if ANIMAL_YOLO_DIR not in sys.path:
    sys.path.insert(0, ANIMAL_YOLO_DIR)

import func as rknn_func  # noqa: E402
from func import IMG_SIZE as RKNN_IMG_SIZE  # noqa: E402
from func import sigmoid, yolov5_post_process  # noqa: E402
from rknnpool import initRKNN  # noqa: E402


DEFAULT_CLASSES = ["tiger", "peacock", "monkey", "elephant", "wolf"]

g_pose_lock = threading.Lock()
g_current_pose = None
g_vision_enabled = True
g_gate_dirty = True


def get_detector_param(name, default):
    private_name = "~{}".format(name)
    if rospy.has_param(private_name):
        return rospy.get_param(private_name)

    # Mission_controller.launch runs this node under /Drone_2, while mission_controller.yaml is
    # loaded at the root. This fallback keeps those global detector params usable.
    global_name = "/animal_detector/{}".format(name)
    if rospy.has_param(global_name):
        return rospy.get_param(global_name)
    return default


def repair_logging_levels():
    # ROS Python logging reads level names during init_node(); mixed Python
    # environments can leave these maps incomplete.
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


def get_default_model_path():
    candidates = [
        os.path.join(ANIMAL_YOLO_DIR, "rknnModel", "yolov5_animal_5class.rknn"),
        os.path.join(ANIMAL_YOLO_DIR, "rknnModel", "yolov5s_relu_tk2_RK3588_i8.rknn"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


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


def build_tracking_info(detections, image_center):
    tracking = {
        "target_count": len(detections),
        "avg_dx": 0.0,
        "avg_dy": 0.0,
        "primary_dx": 0.0,
        "primary_dy": 0.0,
        "primary_label": "",
        "primary_score": 0.0,
        "primary_area_px": 0.0,
        "aim_px_x": 0,
        "aim_px_y": 0,
        "valid": False,
    }
    if not detections:
        return tracking

    diffs_x = []
    diffs_y = []
    primary_detection = None
    primary_rank = None
    for detection in detections:
        bx1, by1, bx2, by2 = detection["box"]
        width = float(bx2 - bx1)
        height = float(by2 - by1)
        cx = int((bx1 + bx2) / 2.0)
        cy = int(by1 + 0.62 * height)
        dx = float(cx - image_center[0])
        dy = float(cy - image_center[1])
        area = float(width * height)
        diffs_x.append(dx)
        diffs_y.append(dy)

        rank = (float(detection["score"]), area)
        if primary_rank is None or rank > primary_rank:
            primary_rank = rank
            primary_detection = {
                "label": detection["label"],
                "score": float(detection["score"]),
                "area_px": area,
                "aim_px": (cx, cy),
                "dx": dx,
                "dy": dy,
            }

    tracking["avg_dx"] = round(sum(diffs_x) / float(len(diffs_x)), 3)
    tracking["avg_dy"] = round(sum(diffs_y) / float(len(diffs_y)), 3)
    tracking["valid"] = primary_detection is not None
    if primary_detection is not None:
        tracking["primary_dx"] = round(primary_detection["dx"], 3)
        tracking["primary_dy"] = round(primary_detection["dy"], 3)
        tracking["primary_label"] = primary_detection["label"]
        tracking["primary_score"] = round(primary_detection["score"], 4)
        tracking["primary_area_px"] = round(primary_detection["area_px"], 1)
        tracking["aim_px_x"] = int(primary_detection["aim_px"][0])
        tracking["aim_px_y"] = int(primary_detection["aim_px"][1])
    return tracking


def build_summary(class_names, counts, diffs, tracking):
    total = sum(int(counts.get(name, 0)) for name in class_names)
    return {
        "timestamp": round(rospy.Time.now().to_sec(), 3),
        "source": "rknn",
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


def publish_empty_summary(class_names, detection_pub, control_pub):
    counts = {name: 0 for name in class_names}
    return publish_summary(
        build_summary(class_names, counts, [], build_tracking_info([], (0, 0))),
        class_names,
        detection_pub,
        control_pub)


def build_center_region(frame_width, frame_height, width_ratio, height_ratio):
    x1 = int(frame_width * (1.0 - width_ratio) / 2.0)
    y1 = int(frame_height * (1.0 - height_ratio) / 2.0)
    x2 = frame_width - x1
    y2 = frame_height - y1
    return x1, y1, x2, y2


def center_contains(box, center_region):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return center_region[0] <= cx <= center_region[2] and center_region[1] <= cy <= center_region[3]


def prepare_model_outputs(outputs, outputs_already_sigmoid):
    if outputs_already_sigmoid:
        return outputs[0], outputs[1], outputs[2]
    return sigmoid(outputs[0]), sigmoid(outputs[1]), sigmoid(outputs[2])


def infer_detections(rknn_lite, frame, class_names, outputs_already_sigmoid):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (RKNN_IMG_SIZE, RKNN_IMG_SIZE))
    outputs = rknn_lite.inference(inputs=[np.expand_dims(resized, axis=0)])

    output0, output1, output2 = prepare_model_outputs(outputs, outputs_already_sigmoid)

    input0 = output0.reshape([3, -1] + list(output0.shape[-2:]))
    input1 = output1.reshape([3, -1] + list(output1.shape[-2:]))
    input2 = output2.reshape([3, -1] + list(output2.shape[-2:]))
    input_data = [
        np.transpose(input0, (2, 3, 0, 1)),
        np.transpose(input1, (2, 3, 0, 1)),
        np.transpose(input2, (2, 3, 0, 1)),
    ]

    boxes, classes, scores = yolov5_post_process(input_data)
    if boxes is None or classes is None or scores is None:
        return []

    scale_x = float(frame.shape[1]) / float(RKNN_IMG_SIZE)
    scale_y = float(frame.shape[0]) / float(RKNN_IMG_SIZE)
    detections = []
    for box, class_id, score in zip(boxes, classes, scores):
        if int(class_id) < 0 or int(class_id) >= len(class_names):
            continue
        x1, y1, x2, y2 = box
        x1 = int(round(float(x1) * scale_x))
        y1 = int(round(float(y1) * scale_y))
        x2 = int(round(float(x2) * scale_x))
        y2 = int(round(float(y2) * scale_y))
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "label": class_names[int(class_id)],
                "score": float(score),
            }
        )
    return detections


def draw_overlay(frame, detections, center_region, paused, summary_text):
    x1, y1, x2, y2 = center_region
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0.0, frame)

    image_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    cv2.drawMarker(frame, image_center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    if paused:
        cv2.putText(frame, "DETECTION PAUSED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if summary_text:
        cv2.putText(frame, summary_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    for detection in detections:
        bx1, by1, bx2, by2 = detection["box"]
        cx = int((bx1 + bx2) / 2.0)
        cy = int((by1 + by2) / 2.0)
        dx = cx - image_center[0]
        dy = cy - image_center[1]
        label = "{} {:.2f} dx={} dy={}".format(detection["label"], detection["score"], dx, dy)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, label, (bx1, max(18, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return frame


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
    global g_gate_dirty

    repair_logging_levels()
    rospy.init_node("animal_detector")

    model_path = get_detector_param("model_path", "")
    if not model_path:
        model_path = get_default_model_path()
    if not os.path.isfile(model_path):
        rospy.logerr("animal_detector: RKNN model not found: %s", model_path)
        return

    class_names = get_detector_param("classes", DEFAULT_CLASSES)
    if not class_names:
        class_names = list(DEFAULT_CLASSES)

    camera_source = normalize_camera_source(get_detector_param("camera_id", 0))
    frame_width = int(get_detector_param("frame_width", 640))
    frame_height = int(get_detector_param("frame_height", 480))
    region_width_ratio = float(get_detector_param("region_width_ratio", 0.45))
    region_height_ratio = float(get_detector_param("region_height_ratio", 0.60))
    min_area = float(get_detector_param("min_area", 500.0))
    max_area = float(get_detector_param("max_area", 25000.0))
    pose_topic = get_detector_param("pose_topic", "mavros/local_position/pose")
    show_window = bool(get_detector_param("show_window", False))
    publish_raw_image = bool(get_detector_param("publish_raw_image", True))
    raw_image_topic = get_detector_param("raw_image_topic", "vision/raw_image")
    raw_frame_id = get_detector_param("raw_frame_id", "animal_detector_camera")
    publish_debug_image = bool(get_detector_param("publish_debug_image", True))
    debug_image_topic = get_detector_param("debug_image_topic", "vision/detection_image")
    debug_frame_id = get_detector_param("debug_frame_id", "animal_detector_camera")
    loop_hz = float(get_detector_param("loop_hz", 15.0))
    npu_core = int(get_detector_param("npu_core", -1))
    obj_thresh = float(get_detector_param("obj_thresh", rknn_func.OBJ_THRESH))
    nms_thresh = float(get_detector_param("nms_thresh", rknn_func.NMS_THRESH))
    score_thresh = float(get_detector_param("score_thresh", 0.50))
    outputs_already_sigmoid = bool(get_detector_param("outputs_already_sigmoid", True))

    rknn_func.OBJ_THRESH = obj_thresh
    rknn_func.NMS_THRESH = nms_thresh

    rospy.Subscriber(pose_topic, PoseStamped, pose_cb, queue_size=10)
    rospy.Subscriber("vision/check", Bool, vision_check_cb, queue_size=10)
    detection_pub = rospy.Publisher("vision/detections_json", String, queue_size=10)
    control_pub = rospy.Publisher("vision/control", Bool, queue_size=10)
    raw_image_pub = rospy.Publisher(raw_image_topic, Image, queue_size=1) if publish_raw_image else None
    debug_image_pub = rospy.Publisher(debug_image_topic, Image, queue_size=1) if publish_debug_image else None

    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        rospy.logerr("animal_detector: failed to open camera %s", str(camera_source))
        return
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    rknn_lite = initRKNN(model_path, npu_core)
    publish_empty_summary(class_names, detection_pub, control_pub)
    rospy.loginfo(
        "animal_detector: ready model=%s camera=%s obj=%.2f score=%.2f nms=%.2f outputs_already_sigmoid=%s",
        model_path,
        str(camera_source),
        obj_thresh,
        score_thresh,
        nms_thresh,
        "true" if outputs_already_sigmoid else "false",
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
                rospy.logwarn_throttle(2.0, "animal_detector: failed to read camera frame")
                rate.sleep()
                continue

            if publish_raw_image:
                raw_image_pub.publish(cv2_to_image_msg(frame, raw_frame_id))

            center_region = build_center_region(frame.shape[1], frame.shape[0], region_width_ratio, region_height_ratio)
            detections = []
            if g_vision_enabled:
                raw_detections = infer_detections(rknn_lite, frame, class_names, outputs_already_sigmoid)
                for detection in raw_detections:
                    if float(detection["score"]) < score_thresh:
                        continue
                    bx1, by1, bx2, by2 = detection["box"]
                    area = float((bx2 - bx1) * (by2 - by1))
                    if area < min_area or area > max_area:
                        continue
                    if not center_contains(detection["box"], center_region):
                        continue
                    detections.append(detection)

                counts = {name: 0 for name in class_names}
                diffs = []
                image_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                for detection in detections:
                    counts[detection["label"]] += 1
                    bx1, by1, bx2, by2 = detection["box"]
                    height = float(by2 - by1)
                    cx = int((bx1 + bx2) / 2.0)
                    cy = int(by1 + 0.62 * height)
                    diffs.append(
                        {
                            "class": detection["label"],
                            "dx": round(float(cx - image_center[0]), 1),
                            "dy": round(float(cy - image_center[1]), 1),
                            "score": round(float(detection["score"]), 4),
                            "area_px": round(float((bx2 - bx1) * (by2 - by1)), 1),
                        }
                    )

                tracking = build_tracking_info(detections, image_center)
                publish_summary(
                    build_summary(class_names, counts, diffs, tracking),
                    class_names,
                    detection_pub,
                    control_pub)
                last_detections = detections
                last_summary_text = " ".join(
                    "{}:{}".format(name, counts[name]) for name in class_names if counts[name] > 0
                )
            else:
                control_pub.publish(Bool(data=False))

            if show_window or publish_debug_image:
                output_detections = last_detections if g_vision_enabled else []
                output = draw_overlay(frame.copy(), output_detections, center_region, not g_vision_enabled, last_summary_text)
                if publish_debug_image:
                    debug_image_pub.publish(cv2_to_image_msg(output, debug_frame_id))
                if not show_window:
                    rate.sleep()
                    continue
                cv2.imshow("RKNN Animal Detection", output)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            rate.sleep()
    finally:
        capture.release()
        if show_window:
            cv2.destroyAllWindows()
        rknn_lite.release()


if __name__ == "__main__":
    main()
