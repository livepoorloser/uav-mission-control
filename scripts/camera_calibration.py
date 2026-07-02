#!/usr/bin/env python3
# 功能：相机标定工具，计算相机内参和畸变参数。

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np


def parse_camera_source(value):
    text = str(value)
    if text.isdigit():
        return int(text)
    return text


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def make_object_points(board_cols, board_rows, square_size):
    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def find_chessboard_corners(gray, board_cols, board_rows):
    pattern_size = (board_cols, board_rows)

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
            if ok:
                return True, corners.astype(np.float32)
        except cv2.error:
            pass

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def put_status(image, lines):
    y = 28
    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26


def save_frame(output_dir, prefix, frame, index):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "{}_{:04d}.jpg".format(prefix, index))
    ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("failed to write {}".format(path))
    return path


def capture_images(args):
    camera_source = parse_camera_source(args.camera)
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        raise RuntimeError("failed to open camera {}".format(args.camera))

    if args.width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ensure_dir(args.output)
    saved = 0
    last_auto_save = 0.0
    print("camera_calibration: capture started")
    print("  board inner corners: {}x{}".format(args.board_cols, args.board_rows))
    print("  output: {}".format(os.path.abspath(args.output)))
    print("  keys: SPACE/s save detected frame, c save raw frame, q/ESC quit")

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            print("warning: failed to read camera frame")
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_chessboard_corners(gray, args.board_cols, args.board_rows)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(
                display,
                (args.board_cols, args.board_rows),
                corners,
                found,
            )

        now = time.time()
        if args.auto and found and now - last_auto_save >= args.auto_interval:
            saved += 1
            path = save_frame(args.output, args.prefix, frame, saved)
            print("saved {} ({}/{})".format(path, saved, args.max_images))
            last_auto_save = now

        put_status(
            display,
            [
                "board: {}x{} inner corners".format(args.board_cols, args.board_rows),
                "detected: {}  saved: {}".format("yes" if found else "no", saved),
                "SPACE/s save detected, c save raw, q quit",
            ],
        )
        cv2.imshow("camera calibration capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord(" "), ord("s")):
            if not found:
                print("not saved: chessboard not detected")
                continue
            saved += 1
            path = save_frame(args.output, args.prefix, frame, saved)
            print("saved {} ({}/{})".format(path, saved, args.max_images))
        elif key == ord("c"):
            saved += 1
            path = save_frame(args.output, args.prefix, frame, saved)
            print("saved raw {} ({}/{})".format(path, saved, args.max_images))

        if args.max_images > 0 and saved >= args.max_images:
            break

    capture.release()
    cv2.destroyAllWindows()
    print("camera_calibration: captured {} images".format(saved))


def expand_image_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        elif os.path.isfile(pattern):
            paths.append(pattern)
    return sorted(set(paths))


def write_opencv_yaml(path, result):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError("failed to open {} for writing".format(path))

    fs.write("image_width", int(result["image_width"]))
    fs.write("image_height", int(result["image_height"]))
    fs.write("board_cols", int(result["board_cols"]))
    fs.write("board_rows", int(result["board_rows"]))
    fs.write("square_size", float(result["square_size"]))
    fs.write("rms_reprojection_error", float(result["rms"]))
    fs.write("mean_per_view_error", float(result["mean_per_view_error"]))
    fs.write("camera_matrix", np.array(result["camera_matrix"], dtype=np.float64))
    fs.write(
        "distortion_coefficients",
        np.array(result["distortion_coefficients"], dtype=np.float64),
    )
    fs.write("new_camera_matrix", np.array(result["new_camera_matrix"], dtype=np.float64))
    fs.write("roi", np.array(result["roi"], dtype=np.int32))
    fs.write("per_view_errors", np.array(result["per_view_errors"], dtype=np.float64))
    fs.release()


def save_json(path, result):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def calibrate_from_images(args):
    image_paths = expand_image_paths(args.images)
    if not image_paths:
        raise RuntimeError("no images matched: {}".format(" ".join(args.images)))

    object_template = make_object_points(args.board_cols, args.board_rows, args.square_size)
    object_points = []
    image_points = []
    accepted = []
    rejected = []
    image_size = None

    ensure_dir(args.annotated_dir)

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"path": path, "reason": "failed_to_read"})
            continue

        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            rejected.append({"path": path, "reason": "image_size_mismatch"})
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = find_chessboard_corners(gray, args.board_cols, args.board_rows)
        if not found:
            rejected.append({"path": path, "reason": "corners_not_found"})
            continue

        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append({"path": path, "corners": corners})

        if args.annotated_dir:
            annotated = image.copy()
            cv2.drawChessboardCorners(
                annotated,
                (args.board_cols, args.board_rows),
                corners,
                True,
            )
            cv2.imwrite(
                os.path.join(args.annotated_dir, os.path.basename(path)),
                annotated,
            )

    if len(accepted) < args.min_views:
        raise RuntimeError(
            "only {} valid views found; need at least {}".format(
                len(accepted), args.min_views
            )
        )

    flags = 0
    if args.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    per_view_errors = []
    for objp, corners, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(corners, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        per_view_errors.append(float(error))

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        args.alpha,
        image_size,
    )

    ensure_dir(args.undistorted_dir)
    if args.undistorted_dir:
        for item in accepted:
            image = cv2.imread(item["path"], cv2.IMREAD_COLOR)
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
            cv2.imwrite(
                os.path.join(args.undistorted_dir, os.path.basename(item["path"])),
                undistorted,
            )

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "board_cols": args.board_cols,
        "board_rows": args.board_rows,
        "square_size": args.square_size,
        "valid_views": len(accepted),
        "rejected_views": len(rejected),
        "rms": float(rms),
        "mean_per_view_error": float(np.mean(per_view_errors)),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1, 1).tolist(),
        "new_camera_matrix": new_camera_matrix.tolist(),
        "roi": [int(v) for v in roi],
        "per_view_errors": per_view_errors,
        "accepted_images": [item["path"] for item in accepted],
        "rejected_images": rejected,
    }

    ensure_dir(os.path.dirname(os.path.abspath(args.output)))
    write_opencv_yaml(args.output, result)
    if args.json_output:
        save_json(args.json_output, result)

    print("camera_calibration: calibration complete")
    print("  valid views: {}".format(len(accepted)))
    print("  rejected views: {}".format(len(rejected)))
    print("  RMS reprojection error: {:.4f} px".format(rms))
    print("  mean per-view error: {:.4f} px".format(np.mean(per_view_errors)))
    print("  camera matrix:")
    print(camera_matrix)
    print("  distortion coefficients:")
    print(dist_coeffs.reshape(-1))
    print("  wrote {}".format(os.path.abspath(args.output)))
    if args.json_output:
        print("  wrote {}".format(os.path.abspath(args.json_output)))


def add_board_args(parser):
    parser.add_argument(
        "--board-cols",
        type=int,
        required=True,
        help="Number of inner chessboard corners along the board width.",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        required=True,
        help="Number of inner chessboard corners along the board height.",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="OpenCV chessboard camera calibration tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="Capture chessboard images from a camera.")
    add_board_args(capture_parser)
    capture_parser.add_argument("--camera", default="0", help="Camera index, video path, or stream URL.")
    capture_parser.add_argument("--width", type=int, default=0, help="Requested capture width. 0 keeps default.")
    capture_parser.add_argument("--height", type=int, default=0, help="Requested capture height. 0 keeps default.")
    capture_parser.add_argument("--output", default="calibration_images", help="Directory for captured images.")
    capture_parser.add_argument("--prefix", default="calib", help="Captured image filename prefix.")
    capture_parser.add_argument("--max-images", type=int, default=30, help="Stop after this many saved images. 0 disables the limit.")
    capture_parser.add_argument("--auto", action="store_true", help="Automatically save frames when the chessboard is detected.")
    capture_parser.add_argument("--auto-interval", type=float, default=1.0, help="Minimum seconds between auto-saved frames.")
    capture_parser.set_defaults(func=capture_images)

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate from saved chessboard images.")
    add_board_args(calibrate_parser)
    calibrate_parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Chessboard square side length, in your preferred unit such as meters.",
    )
    calibrate_parser.add_argument(
        "--images",
        nargs="+",
        default=["calibration_images/*.jpg", "calibration_images/*.png"],
        help="Input image paths or glob patterns.",
    )
    calibrate_parser.add_argument("--output", default="camera_calibration.yaml", help="OpenCV YAML output path.")
    calibrate_parser.add_argument("--json-output", default="camera_calibration.json", help="JSON output path. Empty disables JSON.")
    calibrate_parser.add_argument("--annotated-dir", default="", help="Optional directory for corner visualization images.")
    calibrate_parser.add_argument("--undistorted-dir", default="", help="Optional directory for undistorted preview images.")
    calibrate_parser.add_argument("--alpha", type=float, default=0.0, help="OpenCV new camera matrix alpha, 0 crops invalid pixels, 1 keeps all pixels.")
    calibrate_parser.add_argument("--min-views", type=int, default=8, help="Minimum valid chessboard images required.")
    calibrate_parser.add_argument("--rational-model", action="store_true", help="Enable OpenCV CALIB_RATIONAL_MODEL.")
    calibrate_parser.add_argument("--fix-k3", action="store_true", help="Fix k3 during calibration.")
    calibrate_parser.set_defaults(func=calibrate_from_images)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "json_output") and args.json_output == "":
        args.json_output = None
    args.func(args)


if __name__ == "__main__":
    main()
