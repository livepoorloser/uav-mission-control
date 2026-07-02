#!/usr/bin/env python3
# 功能：对文件夹图片批量运行 RKNN 推理。


import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent
ANIMAL_YOLO_DIR = WORKSPACE_DIR / "animal_yolov5"

if str(ANIMAL_YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_YOLO_DIR))

import func as rknn_func  # noqa: E402
from func import IMG_SIZE, sigmoid, yolov5_post_process  # noqa: E402
from rknnpool import initRKNN  # noqa: E402


DEFAULT_CLASSES = ["tiger", "peacock", "monkey", "elephant", "wolf"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_classes(value):
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise argparse.ArgumentTypeError("classes must contain at least one class name")
    return classes


def iter_images(source):
    source_path = Path(source)
    if source_path.is_file():
        if source_path.suffix.lower() in IMAGE_SUFFIXES:
            yield source_path
        return

    for path in sorted(source_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def run_rknn_image(rknn_lite, image):
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)
    outputs = rknn_lite.inference(inputs=[input_data])

    output0 = sigmoid(outputs[0])
    output1 = sigmoid(outputs[1])
    output2 = sigmoid(outputs[2])

    input0_data = output0.reshape([3, -1] + list(output0.shape[-2:]))
    input1_data = output1.reshape([3, -1] + list(output1.shape[-2:]))
    input2_data = output2.reshape([3, -1] + list(output2.shape[-2:]))

    post_inputs = [
        np.transpose(input0_data, (2, 3, 0, 1)),
        np.transpose(input1_data, (2, 3, 0, 1)),
        np.transpose(input2_data, (2, 3, 0, 1)),
    ]
    return yolov5_post_process(post_inputs)


def box_overlap_metrics(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0.0 else 0.0
    min_area = min(area_a, area_b)
    containment = intersection / min_area if min_area > 0.0 else 0.0
    return iou, containment


def box_center_distance_sq(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    return (acx - bcx) * (acx - bcx) + (acy - bcy) * (acy - bcy)


def suppress_duplicate_predictions(boxes, class_ids, scores, iou_thresh, center_px, containment_thresh):
    if boxes is None or class_ids is None or scores is None:
        return boxes, class_ids, scores

    center_thresh_sq = float(center_px) * float(center_px)
    keep = []
    for idx in np.argsort(scores)[::-1]:
        duplicate = False
        for kept_idx in keep:
            iou, containment = box_overlap_metrics(boxes[idx], boxes[kept_idx])
            overlap_match = float(iou_thresh) > 0.0 and iou >= float(iou_thresh)
            containment_match = float(containment_thresh) > 0.0 and containment >= float(containment_thresh)
            center_match = float(center_px) > 0.0 and box_center_distance_sq(boxes[idx], boxes[kept_idx]) <= center_thresh_sq
            if overlap_match or containment_match or center_match:
                duplicate = True
                break
        if not duplicate:
            keep.append(idx)

    keep = np.array(keep, dtype=np.int64)
    return boxes[keep], class_ids[keep], scores[keep]


def draw_detection(image, box, score, class_id, class_names):
    h, w = image.shape[:2]
    x_scale = float(w) / float(IMG_SIZE)
    y_scale = float(h) / float(IMG_SIZE)
    x1, y1, x2, y2 = box
    x1 = int(round(x1 * x_scale))
    y1 = int(round(y1 * y_scale))
    x2 = int(round(x2 * x_scale))
    y2 = int(round(y2 * y_scale))

    label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        "{} {:.2f}".format(label, score),
        (max(0, x1), max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    parser = argparse.ArgumentParser(description="Run an RKNN YOLOv5 model on a folder of images.")
    parser.add_argument("--model", required=True, help="Path to .rknn model")
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--output-dir", default="rknn_check_output", help="Directory for annotated images and CSV")
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=DEFAULT_CLASSES,
        help="Comma-separated class names in model order",
    )
    parser.add_argument("--npu-core", type=int, default=-1, help="-1 uses all RK3588 NPU cores")
    parser.add_argument("--obj-thresh", type=float, default=0.50, help="Object/class threshold used by RKNN YOLO postprocess")
    parser.add_argument("--score-thresh", type=float, default=0.50, help="Final confidence threshold after objectness * class score")
    parser.add_argument("--nms-thresh", type=float, default=0.45, help="Same-class NMS IoU threshold")
    parser.add_argument("--duplicate-iou-thresh", type=float, default=0.30, help="Class-agnostic duplicate IoU threshold")
    parser.add_argument("--duplicate-center-px", type=float, default=55.0, help="Class-agnostic duplicate center distance threshold")
    parser.add_argument(
        "--duplicate-containment-thresh",
        type=float,
        default=0.65,
        help="Class-agnostic duplicate containment threshold",
    )
    args = parser.parse_args()

    rknn_func.OBJ_THRESH = args.obj_thresh
    rknn_func.NMS_THRESH = args.nms_thresh

    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"

    rknn_lite = initRKNN(args.model, args.npu_core)
    rows = []
    class_counts = {name: 0 for name in args.classes}

    try:
        for image_path in iter_images(args.source):
            image = cv2.imread(str(image_path))
            if image is None:
                print("skip unreadable image:", image_path)
                continue

            boxes, class_ids, scores = run_rknn_image(rknn_lite, image)
            if boxes is not None:
                keep = scores >= args.score_thresh
                boxes = boxes[keep]
                class_ids = class_ids[keep]
                scores = scores[keep]
            boxes, class_ids, scores = suppress_duplicate_predictions(
                boxes,
                class_ids,
                scores,
                args.duplicate_iou_thresh,
                args.duplicate_center_px,
                args.duplicate_containment_thresh,
            )
            annotated = image.copy()
            if boxes is not None:
                for box, class_id, score in zip(boxes, class_ids, scores):
                    class_id = int(class_id)
                    score = float(score)
                    label = args.classes[class_id] if 0 <= class_id < len(args.classes) else str(class_id)
                    class_counts[label] = class_counts.get(label, 0) + 1
                    rows.append(
                        {
                            "image": str(image_path),
                            "class_id": class_id,
                            "class_name": label,
                            "score": "{:.6f}".format(score),
                            "x1": "{:.2f}".format(float(box[0])),
                            "y1": "{:.2f}".format(float(box[1])),
                            "x2": "{:.2f}".format(float(box[2])),
                            "y2": "{:.2f}".format(float(box[3])),
                        }
                    )
                    draw_detection(annotated, box, score, class_id, args.classes)
            else:
                rows.append(
                    {
                        "image": str(image_path),
                        "class_id": "",
                        "class_name": "",
                        "score": "",
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                    }
                )

            cv2.imwrite(str(annotated_dir / image_path.name), annotated)
    finally:
        rknn_lite.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    print("wrote:", csv_path)
    print("annotated:", annotated_dir)
    print("class counts:")
    for name, count in class_counts.items():
        print("  {}: {}".format(name, count))


if __name__ == "__main__":
    main()
