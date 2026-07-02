#!/usr/bin/env python3
# 功能：对视觉伺服日志做频域分析，辅助判断振荡问题。

import argparse
import csv
import math


def load_columns(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("CSV has no data rows")
    out = {}
    for key in rows[0].keys():
        out[key] = []
    for row in rows:
        for key, value in row.items():
            out[key].append(value)
    return out


def to_float_list(values):
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(float("nan"))
    return result


def interpolate_uniform(times, values, sample_hz):
    start = times[0]
    end = times[-1]
    if end <= start:
        raise RuntimeError("time column is not increasing")
    dt = 1.0 / sample_hz
    n = int(math.floor((end - start) / dt)) + 1
    uniform_t = [start + i * dt for i in range(n)]
    uniform_v = []
    j = 0
    for t in uniform_t:
        while j + 1 < len(times) and times[j + 1] < t:
            j += 1
        if j + 1 >= len(times):
            uniform_v.append(values[-1])
            continue
        t0, t1 = times[j], times[j + 1]
        v0, v1 = values[j], values[j + 1]
        if t1 <= t0:
            uniform_v.append(v0)
        else:
            alpha = (t - t0) / (t1 - t0)
            uniform_v.append(v0 + alpha * (v1 - v0))
    return uniform_t, uniform_v


def dft_power(values, sample_hz):
    n = len(values)
    mean = sum(values) / n
    centered = [v - mean for v in values]
    result = []
    for k in range(1, n // 2 + 1):
        re = 0.0
        im = 0.0
        for i, value in enumerate(centered):
            angle = -2.0 * math.pi * k * i / n
            re += value * math.cos(angle)
            im += value * math.sin(angle)
        freq = k * sample_hz / n
        amp = 2.0 * math.sqrt(re * re + im * im) / n
        result.append((freq, amp))
    return result


def summarize_axis(name, times, values):
    finite = [(t, v) for t, v in zip(times, values) if math.isfinite(t) and math.isfinite(v)]
    if len(finite) < 6:
        print("{}: not enough samples".format(name))
        return
    times = [item[0] for item in finite]
    values = [item[1] for item in finite]
    duration = times[-1] - times[0]
    if duration <= 0.0:
        print("{}: invalid duration".format(name))
        return
    raw_hz = (len(times) - 1) / duration
    sample_hz = max(2.0, min(20.0, raw_hz * 2.0))
    _, uniform_values = interpolate_uniform(times, values, sample_hz)
    spectrum = dft_power(uniform_values, sample_hz)
    spectrum = [(freq, amp) for freq, amp in spectrum if freq > 0.03]
    spectrum.sort(key=lambda item: item[1], reverse=True)
    mean = sum(values) / len(values)
    rms = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    print("{}: samples={} duration={:.2f}s raw_hz={:.2f} mean={:.2f} rms={:.2f}".format(
        name, len(values), duration, raw_hz, mean, rms))
    for freq, amp in spectrum[:5]:
        period = 1.0 / freq if freq > 0.0 else float("inf")
        print("  peak freq={:.3f}Hz period={:.2f}s amp={:.2f}".format(freq, period, amp))


def main():
    parser = argparse.ArgumentParser(description="Analyze vision servo CSV with a simple FFT.")
    parser.add_argument("csv_path")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["error_x_px", "error_y_px", "servo_error_x_px", "servo_error_y_px", "pose_x", "pose_y"],
        help="CSV columns to analyze")
    args = parser.parse_args()

    columns = load_columns(args.csv_path)
    times = to_float_list(columns["time"])
    if times:
        t0 = times[0]
        times = [t - t0 for t in times]
    for column in args.columns:
        if column not in columns:
            print("{}: missing".format(column))
            continue
        summarize_axis(column, times, to_float_list(columns[column]))


if __name__ == "__main__":
    main()
