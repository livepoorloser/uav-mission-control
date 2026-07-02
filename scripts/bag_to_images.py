#!/usr/bin/env python3
# 功能：从 ROS bag 中提取图像帧用于数据检查或标注。

import argparse
import os

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge, CvBridgeError


def safe_topic_name(topic):
    return topic.strip("/").replace("/", "_") or "root"


def image_from_msg(bridge, msg):
    msg_type = getattr(msg, "_type", "")
    if msg_type == "sensor_msgs/CompressedImage":
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode compressed image")
        return image

    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from ROS bag image topics.")
    parser.add_argument("bag", help="Input .bag file")
    parser.add_argument(
        "-o", "--output", default="bag_images", help="Output directory")
    parser.add_argument(
        "-t", "--topic", action="append", default=[],
        help="Image topic to export. Can be used multiple times.")
    parser.add_argument(
        "--every-n", type=int, default=1,
        help="Save one image every N messages from each topic.")
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Stop after this many images per topic. 0 means no limit.")
    parser.add_argument(
        "--format", choices=["jpg", "png"], default="jpg",
        help="Output image format")
    parser.add_argument(
        "--jpg-quality", type=int, default=95,
        help="JPEG quality, 1-100")
    args = parser.parse_args()

    topics = args.topic or ["/vision/raw_image"]
    os.makedirs(args.output, exist_ok=True)

    bridge = CvBridge()
    seen = {topic: 0 for topic in topics}
    saved = {topic: 0 for topic in topics}

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=topics):
            seen[topic] += 1
            if args.every_n > 1 and (seen[topic] - 1) % args.every_n != 0:
                continue
            if args.max_images > 0 and saved[topic] >= args.max_images:
                continue

            try:
                image = image_from_msg(bridge, msg)
            except Exception as exc:
                print("skip {} at {}: {}".format(topic, stamp.to_sec(), exc))
                continue

            topic_dir = os.path.join(args.output, safe_topic_name(topic))
            os.makedirs(topic_dir, exist_ok=True)
            filename = "{:06d}_{:.6f}.{}".format(
                saved[topic], stamp.to_sec(), args.format)
            path = os.path.join(topic_dir, filename)

            if args.format == "jpg":
                cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
            else:
                cv2.imwrite(path, image)
            saved[topic] += 1

    for topic in topics:
        print("{}: seen={}, saved={}".format(topic, seen[topic], saved[topic]))


if __name__ == "__main__":
    main()
