#!/usr/bin/env python3
# 功能：从 ROS bag 中分析视觉伺服数据并导出指标。

import argparse
import bisect
import csv
import json
import math
import os
import sys

try:
    import rosbag
except ImportError:
    rosbag = None


def stamp_to_sec(stamp):
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def bag_time_to_sec(t):
    return float(t.to_sec())


def norm2(x, y):
    if not math.isfinite(x) or not math.isfinite(y):
        return float("nan")
    return math.hypot(x, y)


def finite(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def choose_topic(available, preferred, suffix):
    if preferred in available:
        return preferred
    matches = sorted(topic for topic in available if topic.endswith(suffix))
    return matches[0] if matches else preferred


def nearest_sample(series, t, max_dt):
    if not series:
        return None
    times = [item["t"] for item in series]
    index = bisect.bisect_left(times, t)
    candidates = []
    if index < len(series):
        candidates.append(series[index])
    if index > 0:
        candidates.append(series[index - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(item["t"] - t))
    if abs(best["t"] - t) > max_dt:
        return None
    return best


def parse_detection(msg, t):
    try:
        payload = json.loads(msg.data)
    except (TypeError, ValueError):
        return None

    tracking = payload.get("tracking", {}) or {}
    counts = payload.get("counts", {}) or {}
    dx = finite(tracking.get("primary_dx", tracking.get("avg_dx")))
    dy = finite(tracking.get("primary_dy", tracking.get("avg_dy")))
    valid = bool(tracking.get("valid", False))
    area = finite(tracking.get("primary_area_px", 0.0), 0.0)
    score = finite(tracking.get("primary_score", 0.0), 0.0)
    label = tracking.get("primary_label", "")
    total = finite(payload.get("total", 0.0), 0.0)
    return {
        "t": t,
        "dx": dx,
        "dy": dy,
        "valid": valid,
        "area": area,
        "score": score,
        "label": label,
        "total": total,
        "source": payload.get("source", ""),
        "tiger": int(counts.get("tiger", 0) or 0),
    }


def parse_pose(msg, t):
    p = msg.pose.position
    return {"t": t, "x": finite(p.x), "y": finite(p.y), "z": finite(p.z)}


def parse_twist(msg, t):
    v = msg.twist.linear
    return {"t": t, "vx": finite(v.x), "vy": finite(v.y), "vz": finite(v.z)}


def parse_position_target(msg, t):
    return {
        "t": t,
        "vx": finite(msg.velocity.x),
        "vy": finite(msg.velocity.y),
        "vz": finite(msg.velocity.z),
        "px": finite(msg.position.x),
        "py": finite(msg.position.y),
        "pz": finite(msg.position.z),
        "yaw": finite(msg.yaw),
        "type_mask": int(msg.type_mask),
    }


def parse_state(msg, t):
    return {
        "t": t,
        "mode": msg.mode,
        "armed": bool(msg.armed),
        "connected": bool(msg.connected),
        "guided": bool(msg.guided),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_aligned_rows(detections, poses, velocities, commands, targets, states, args):
    rows = []
    for det in detections:
        t = det["t"]
        pose = nearest_sample(poses, t, args.match_dt) or {}
        vel = nearest_sample(velocities, t, args.match_dt) or {}
        cmd = nearest_sample(commands, t, args.match_dt) or {}
        target = nearest_sample(targets, t, args.match_dt) or {}
        state = nearest_sample(states, t, args.match_dt) or {}

        err_x = det["dx"] - args.target_x
        err_y = det["dy"] - args.target_y
        inside_deadband = (
            det["valid"] and
            abs(err_x) <= args.deadband and
            abs(err_y) <= args.deadband
        )

        cmd_vx = finite(cmd.get("vx"))
        cmd_vy = finite(cmd.get("vy"))
        actual_vx = finite(vel.get("vx"))
        actual_vy = finite(vel.get("vy"))

        rows.append({
            "t": t,
            "dt": 0.0,
            "valid": int(det["valid"]),
            "label": det["label"],
            "source": det["source"],
            "dx_px": det["dx"],
            "dy_px": det["dy"],
            "target_x_px": args.target_x,
            "target_y_px": args.target_y,
            "err_x_px": err_x,
            "err_y_px": err_y,
            "err_norm_px": norm2(err_x, err_y),
            "inside_deadband": int(inside_deadband),
            "area_px": det["area"],
            "score": det["score"],
            "pose_x": finite(pose.get("x")),
            "pose_y": finite(pose.get("y")),
            "pose_z": finite(pose.get("z")),
            "actual_vx": actual_vx,
            "actual_vy": actual_vy,
            "actual_speed_xy": norm2(actual_vx, actual_vy),
            "cmd_vx": cmd_vx,
            "cmd_vy": cmd_vy,
            "cmd_speed_xy": norm2(cmd_vx, cmd_vy),
            "target_vx": finite(target.get("vx")),
            "target_vy": finite(target.get("vy")),
            "mode": state.get("mode", ""),
            "armed": int(bool(state.get("armed", False))),
        })

    if rows:
        t0 = rows[0]["t"]
        for row in rows:
            row["dt"] = row["t"] - t0
    return rows


def summarize(rows, args):
    valid_rows = [row for row in rows if row["valid"]]
    inside_rows = [row for row in valid_rows if row["inside_deadband"]]
    summary = []
    summary.append("# Vision Servo Bag Analysis")
    summary.append("")
    summary.append("## Basic")
    summary.append("")
    summary.append("- samples_total: {}".format(len(rows)))
    summary.append("- samples_valid_detection: {}".format(len(valid_rows)))
    summary.append("- samples_inside_deadband: {}".format(len(inside_rows)))
    summary.append("- target_pixel: ({:.2f}, {:.2f}) px".format(args.target_x, args.target_y))
    summary.append("- deadband: {:.2f} px".format(args.deadband))
    summary.append("")

    if valid_rows:
        best = min(valid_rows, key=lambda row: row["err_norm_px"])
        summary.append("## Closest Target Moment")
        summary.append("")
        summary.append("- time_from_start: {:.2f}s".format(best["dt"]))
        summary.append("- err: ({:.2f}, {:.2f}) px, norm={:.2f}px".format(
            best["err_x_px"], best["err_y_px"], best["err_norm_px"]))
        summary.append("- cmd: ({:.4f}, {:.4f}) m/s".format(best["cmd_vx"], best["cmd_vy"]))
        summary.append("- actual_velocity: ({:.4f}, {:.4f}) m/s, speed={:.4f}m/s".format(
            best["actual_vx"], best["actual_vy"], best["actual_speed_xy"]))
        summary.append("- pose: ({:.3f}, {:.3f}, {:.3f}) m".format(
            best["pose_x"], best["pose_y"], best["pose_z"]))
        summary.append("")

    if inside_rows:
        entry = inside_rows[0]
        later_candidates = [
            row for row in rows
            if row["t"] >= entry["t"] + args.hold_check_s and row["valid"]
        ]
        later = later_candidates[0] if later_candidates else None

        summary.append("## Deadband Entry")
        summary.append("")
        summary.append("- first_entry_time_from_start: {:.2f}s".format(entry["dt"]))
        summary.append("- entry_err: ({:.2f}, {:.2f}) px".format(entry["err_x_px"], entry["err_y_px"]))
        summary.append("- entry_cmd_speed: {:.4f} m/s".format(entry["cmd_speed_xy"]))
        summary.append("- entry_actual_speed: {:.4f} m/s".format(entry["actual_speed_xy"]))

        if later:
            drift_x = later["pose_x"] - entry["pose_x"]
            drift_y = later["pose_y"] - entry["pose_y"]
            err_growth = later["err_norm_px"] - entry["err_norm_px"]
            summary.append("- after_{:.1f}s_err: ({:.2f}, {:.2f}) px, norm={:.2f}px".format(
                args.hold_check_s, later["err_x_px"], later["err_y_px"], later["err_norm_px"]))
            summary.append("- after_{:.1f}s_pose_drift: ({:.3f}, {:.3f}) m, norm={:.3f}m".format(
                args.hold_check_s, drift_x, drift_y, norm2(drift_x, drift_y)))
            summary.append("- err_norm_growth_after_deadband: {:.2f}px".format(err_growth))

            if entry["cmd_speed_xy"] <= args.zero_cmd_speed and entry["actual_speed_xy"] > args.moving_speed:
                summary.append("- diagnosis: cmd is near zero at deadband entry, but actual velocity is still moving.")
            if err_growth > args.release_error:
                summary.append("- diagnosis: error grows again after entering deadband; hold/brake logic is likely insufficient.")
            if norm2(drift_x, drift_y) > args.drift_threshold:
                summary.append("- diagnosis: pose drifts after deadband entry; the aircraft is not physically staying over the target.")
        summary.append("")

    # Saturation and command behavior
    cmd_rows = [row for row in rows if math.isfinite(row["cmd_speed_xy"])]
    if cmd_rows:
        max_cmd = max(row["cmd_speed_xy"] for row in cmd_rows)
        near_zero = [row for row in cmd_rows if row["cmd_speed_xy"] <= args.zero_cmd_speed]
        summary.append("## Command")
        summary.append("")
        summary.append("- max_cmd_speed_xy: {:.4f} m/s".format(max_cmd))
        summary.append("- near_zero_cmd_samples: {}".format(len(near_zero)))
        summary.append("")

    summary.append("## Suggested Interpretation")
    summary.append("")
    summary.append("If the curves show detection error entering the deadband, command speed falling near zero,")
    summary.append("and actual velocity/pose still moving afterward, the issue is not target detection.")
    summary.append("It is mainly the missing hold/brake behavior after reaching the visual target.")
    summary.append("")
    return "\n".join(summary)


def plot_rows(path, rows, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not rows:
        return False

    t = [row["dt"] for row in rows]
    err_x = [row["err_x_px"] if row["valid"] else float("nan") for row in rows]
    err_y = [row["err_y_px"] if row["valid"] else float("nan") for row in rows]
    cmd_vx = [row["cmd_vx"] for row in rows]
    cmd_vy = [row["cmd_vy"] for row in rows]
    actual_vx = [row["actual_vx"] for row in rows]
    actual_vy = [row["actual_vy"] for row in rows]
    pose_x = [row["pose_x"] for row in rows]
    pose_y = [row["pose_y"] for row in rows]
    area = [row["area_px"] if row["valid"] else float("nan") for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, err_x, label="err_x_px")
    axes[0].plot(t, err_y, label="err_y_px")
    axes[0].axhline(args.deadband, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axhline(-args.deadband, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("pixel error")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, cmd_vx, label="cmd_vx")
    axes[1].plot(t, cmd_vy, label="cmd_vy")
    axes[1].plot(t, actual_vx, label="actual_vx", linestyle="--")
    axes[1].plot(t, actual_vy, label="actual_vy", linestyle="--")
    axes[1].set_ylabel("velocity m/s")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(t, pose_x, label="pose_x")
    axes[2].plot(t, pose_y, label="pose_y")
    axes[2].set_ylabel("position m")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    axes[3].plot(t, area, label="target_area_px")
    axes[3].set_ylabel("area px")
    axes[3].set_xlabel("time from first detection sample (s)")
    axes[3].legend(loc="best")
    axes[3].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def read_bag(args):
    if rosbag is None:
        raise RuntimeError("Cannot import rosbag. Run this script in a ROS environment after source devel/setup.bash.")

    detections = []
    poses = []
    velocities = []
    commands = []
    targets = []
    states = []

    with rosbag.Bag(args.bag) as bag:
        available = set(bag.get_type_and_topic_info().topics.keys())
        topic_detection = choose_topic(available, args.topic_detection, "/vision/detections_json")
        topic_pose = choose_topic(available, args.topic_pose, "/mavros/local_position/pose")
        topic_velocity = choose_topic(available, args.topic_velocity, "/mavros/local_position/velocity_local")
        topic_command = choose_topic(available, args.topic_command, "/mavros/setpoint_raw/local")
        topic_target = choose_topic(available, args.topic_target, "/mavros/setpoint_raw/target_local")
        topic_state = choose_topic(available, args.topic_state, "/mavros/state")

        topics = [topic_detection, topic_pose, topic_velocity, topic_command, topic_target, topic_state]
        for topic, msg, stamp in bag.read_messages(topics=topics):
            t = bag_time_to_sec(stamp)
            if topic == topic_detection:
                item = parse_detection(msg, t)
                if item is not None:
                    detections.append(item)
            elif topic == topic_pose:
                poses.append(parse_pose(msg, t))
            elif topic == topic_velocity:
                velocities.append(parse_twist(msg, t))
            elif topic == topic_command:
                commands.append(parse_position_target(msg, t))
            elif topic == topic_target:
                targets.append(parse_position_target(msg, t))
            elif topic == topic_state:
                states.append(parse_state(msg, t))

    for series in [detections, poses, velocities, commands, targets, states]:
        series.sort(key=lambda item: item["t"])

    return detections, poses, velocities, commands, targets, states


def main():
    parser = argparse.ArgumentParser(description="Analyze visual servo rosbag data.")
    parser.add_argument("bag", help="Path to rosbag file.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: <bag>_analysis")
    parser.add_argument("--target-x", type=float, default=-20.0, help="Expected target dx in pixels.")
    parser.add_argument("--target-y", type=float, default=8.8, help="Expected target dy in pixels.")
    parser.add_argument("--deadband", type=float, default=8.0, help="Deadband in pixels used for analysis.")
    parser.add_argument("--match-dt", type=float, default=0.30, help="Max time gap for nearest-topic alignment.")
    parser.add_argument("--hold-check-s", type=float, default=2.0, help="Seconds after deadband entry to check drift.")
    parser.add_argument("--zero-cmd-speed", type=float, default=0.003, help="Command speed treated as zero.")
    parser.add_argument("--moving-speed", type=float, default=0.025, help="Actual speed treated as still moving.")
    parser.add_argument("--drift-threshold", type=float, default=0.05, help="Pose drift threshold after deadband entry.")
    parser.add_argument("--release-error", type=float, default=15.0, help="Error growth threshold after deadband entry.")

    parser.add_argument("--topic-detection", default="/Drone_2/vision/detections_json")
    parser.add_argument("--topic-pose", default="/Drone_2/mavros/local_position/pose")
    parser.add_argument("--topic-velocity", default="/Drone_2/mavros/local_position/velocity_local")
    parser.add_argument("--topic-command", default="/Drone_2/mavros/setpoint_raw/local")
    parser.add_argument("--topic-target", default="/Drone_2/mavros/setpoint_raw/target_local")
    parser.add_argument("--topic-state", default="/Drone_2/mavros/state")

    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir:
        base = os.path.splitext(os.path.basename(args.bag))[0]
        out_dir = os.path.abspath(base + "_analysis")
    ensure_dir(out_dir)

    detections, poses, velocities, commands, targets, states = read_bag(args)
    rows = make_aligned_rows(detections, poses, velocities, commands, targets, states, args)

    csv_path = os.path.join(out_dir, "vision_servo_aligned.csv")
    summary_path = os.path.join(out_dir, "summary.md")
    plot_path = os.path.join(out_dir, "vision_servo_curves.png")

    write_csv(csv_path, rows)
    summary = summarize(rows, args)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    plotted = plot_rows(plot_path, rows, args)

    print("Wrote: {}".format(csv_path))
    print("Wrote: {}".format(summary_path))
    if plotted:
        print("Wrote: {}".format(plot_path))
    else:
        print("Plot skipped: matplotlib is not available or no rows were parsed.")
    print("")
    print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
