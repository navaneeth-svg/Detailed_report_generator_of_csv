#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("pandas is required. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)


def _pick_column(df, preferred, fallbacks):
    if preferred and preferred in df.columns:
        return preferred
    for name in fallbacks:
        if name in df.columns:
            return name
    return None


def _compute_sample_rate(time_s):
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return 1.0 / np.median(dt)


def _detect_segments(time_s, current_a, threshold, min_duration) -> List[Tuple[int, int]]:
    if time_s.size == 0:
        return []
    active = np.abs(current_a) >= threshold
    edges = np.flatnonzero(active[1:] != active[:-1]) + 1

    starts = []
    ends = []

    if active[0]:
        starts.append(0)

    for idx in edges:
        if active[idx]:
            starts.append(idx)
        else:
            ends.append(idx - 1)

    if active[-1]:
        ends.append(len(active) - 1)

    segments = []
    for s, e in zip(starts, ends):
        dur = time_s[e] - time_s[s]
        if dur >= min_duration:
            segments.append((s, e))

    return segments


def _segment_stats(time_s, voltage_v, current_a, seg):
    s, e = seg
    t0, t1 = time_s[s], time_s[e]
    v = voltage_v[s : e + 1]
    i = current_a[s : e + 1]
    return {
        "start_s": float(t0),
        "end_s": float(t1),
        "duration_s": float(t1 - t0),
        "mean_current_a": float(np.mean(i)),
        "peak_current_a": float(np.max(i)),
        "min_current_a": float(np.min(i)),
        "mean_voltage_v": float(np.mean(v)),
        "min_voltage_v": float(np.min(v)),
        "max_voltage_v": float(np.max(v)),
        "delta_v": float(v[-1] - v[0]),
    }


def _classify_segment(mean_current, invert):
    signed = -mean_current if invert else mean_current
    return "discharge" if signed > 0 else "charge"


def _find_spikes(time_s, values, min_rate=None, top_n=6):
    if time_s.size < 2:
        return []
    dt = np.diff(time_s)
    dv = np.diff(values)
    rate = np.zeros_like(dv)
    valid = dt > 0
    rate[valid] = dv[valid] / dt[valid]
    abs_rate = np.abs(rate)

    if min_rate is None:
        med = np.median(abs_rate)
        mad = np.median(np.abs(abs_rate - med))
        min_rate = med + 8 * mad

    idx = np.where(abs_rate >= min_rate)[0]
    if idx.size == 0:
        return []

    order = np.argsort(abs_rate[idx])
    top_idx = idx[order[-top_n:]]

    spikes = []
    for i in top_idx[::-1]:
        spikes.append({
            "time_s": float(time_s[i + 1]),
            "rate": float(rate[i]),
            "abs_rate": float(abs_rate[i]),
            "value": float(values[i + 1]),
        })
    return spikes


def main():
    parser = argparse.ArgumentParser(description="Analyze DAQ CSV and generate a pulse report")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=r"G:\My Drive\Screw_prediction\prediction_daq_20260310_141608_cell2_044_OCV3.915_IR48.86.csv",
        help="Path to DAQ CSV",
    )
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--voltage-col", default=None)
    parser.add_argument("--current-col", default=None)
    parser.add_argument("--invert-current", action="store_true", help="Flip current sign")
    parser.add_argument("--current-threshold", type=float, default=0.2, help="A threshold for pulse detection")
    parser.add_argument("--min-segment", type=float, default=0.1, help="Minimum pulse duration in seconds")
    parser.add_argument("--rest-threshold", type=float, default=0.05, help="A threshold for rest detection")
    parser.add_argument("--baseline-window", type=float, default=1.0, help="Seconds for baseline estimate")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON report")

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    time_col = _pick_column(df, args.time_col, ["Time_s", "Time (s)", "time", "t"])
    if not time_col:
        raise SystemExit("Time column not found. Provide --time-col.")

    voltage_col = _pick_column(df, args.voltage_col, ["Cell_1_V", "Cell_2_V", "Voltage", "V"])
    if not voltage_col:
        raise SystemExit("Voltage column not found. Provide --voltage-col.")

    current_col = _pick_column(
        df,
        args.current_col,
        ["Current_A_Calibrated", "Current_A", "Current", "I"],
    )
    if not current_col:
        raise SystemExit("Current column not found. Provide --current-col.")

    time_s = df[time_col].to_numpy(dtype=float)
    voltage_v = df[voltage_col].to_numpy(dtype=float)
    current_a = df[current_col].to_numpy(dtype=float)

    if current_col == "Current_A":
        # Treat Current_A as sensor voltage if calibrated column is missing
        current_a = (current_a - 2.5) * 10.0
        current_col = "Current_A (calculated)"

    if args.invert_current:
        current_a = -current_a

    sample_rate = _compute_sample_rate(time_s)

    duration_s = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0

    baseline_end = time_s[0] + args.baseline_window
    baseline_mask = time_s <= baseline_end
    rest_mask = np.abs(current_a) <= args.rest_threshold
    baseline_mask = baseline_mask & rest_mask

    if np.any(baseline_mask):
        baseline_v = float(np.median(voltage_v[baseline_mask]))
        baseline_i = float(np.median(current_a[baseline_mask]))
    else:
        baseline_v = float(np.median(voltage_v[: max(1, int(0.1 * len(voltage_v)))]))
        baseline_i = float(np.median(current_a[: max(1, int(0.1 * len(current_a)))]))

    segments = _detect_segments(time_s, current_a, args.current_threshold, args.min_segment)

    rest_segments = []
    prev = 0
    for s, e in segments:
        if s > prev:
            rest_segments.append((prev, s - 1))
        prev = e + 1
    if prev < len(time_s):
        rest_segments.append((prev, len(time_s) - 1))

    segment_details = []
    for seg in segments:
        stats = _segment_stats(time_s, voltage_v, current_a, seg)
        stats["type"] = _classify_segment(stats["mean_current_a"], args.invert_current)
        segment_details.append(stats)

    rest_details = []
    for seg in rest_segments:
        s, e = seg
        if s >= e:
            continue
        t0, t1 = time_s[s], time_s[e]
        if t1 - t0 < args.min_segment:
            continue
        v = voltage_v[s : e + 1]
        i = current_a[s : e + 1]
        rest_details.append({
            "start_s": float(t0),
            "end_s": float(t1),
            "duration_s": float(t1 - t0),
            "mean_current_a": float(np.mean(i)),
            "mean_voltage_v": float(np.mean(v)),
            "min_voltage_v": float(np.min(v)),
            "max_voltage_v": float(np.max(v)),
        })

    current_spikes = _find_spikes(time_s, current_a)
    voltage_spikes = _find_spikes(time_s, voltage_v)

    report = {
        "file": args.csv_path,
        "columns": {
            "time": time_col,
            "voltage": voltage_col,
            "current": current_col,
        },
        "samples": int(len(time_s)),
        "duration_s": duration_s,
        "sample_rate_hz": float(sample_rate) if sample_rate else None,
        "baseline": {
            "window_s": args.baseline_window,
            "voltage_v": baseline_v,
            "current_a": baseline_i,
        },
        "thresholds": {
            "current_threshold_a": args.current_threshold,
            "rest_threshold_a": args.rest_threshold,
            "min_segment_s": args.min_segment,
        },
        "segments": segment_details,
        "rests": rest_details,
        "spikes": {
            "current": current_spikes,
            "voltage": voltage_spikes,
        },
    }

    lines = []
    lines.append(f"File: {report['file']}")
    lines.append(f"Samples: {report['samples']}, Duration: {duration_s:.3f}s")
    if sample_rate:
        lines.append(f"Sample rate: {sample_rate:.1f} Hz")
    lines.append(
        f"Columns: time={time_col}, voltage={voltage_col}, current={current_col}"
    )
    lines.append(
        f"Baseline (first {args.baseline_window}s, rest<= {args.rest_threshold}A): "
        f"V={baseline_v:.3f}V, I={baseline_i:.3f}A"
    )

    lines.append("\nActive segments (|I| >= threshold):")
    if not segment_details:
        lines.append("  None detected")
    else:
        for idx, s in enumerate(segment_details, 1):
            lines.append(
                f"  #{idx} {s['type']} {s['start_s']:.3f}s -> {s['end_s']:.3f}s "
                f"({s['duration_s']:.3f}s), Imean={s['mean_current_a']:.3f}A, "
                f"Ipeak={s['peak_current_a']:.3f}A, Vmean={s['mean_voltage_v']:.3f}V, "
                f"dV={s['delta_v']:.3f}V"
            )

    lines.append("\nRest segments:")
    if not rest_details:
        lines.append("  None detected")
    else:
        for idx, r in enumerate(rest_details, 1):
            lines.append(
                f"  #{idx} {r['start_s']:.3f}s -> {r['end_s']:.3f}s "
                f"({r['duration_s']:.3f}s), Imean={r['mean_current_a']:.3f}A, "
                f"Vmean={r['mean_voltage_v']:.3f}V, Vmin={r['min_voltage_v']:.3f}V, "
                f"Vmax={r['max_voltage_v']:.3f}V"
            )

    lines.append("\nCurrent spikes (dI/dt):")
    if not current_spikes:
        lines.append("  None detected")
    else:
        for s in current_spikes:
            lines.append(
                f"  t={s['time_s']:.3f}s, dI/dt={s['rate']:.2f} A/s, I={s['value']:.3f}A"
            )

    lines.append("\nVoltage spikes (dV/dt):")
    if not voltage_spikes:
        lines.append("  None detected")
    else:
        for s in voltage_spikes:
            lines.append(
                f"  t={s['time_s']:.3f}s, dV/dt={s['rate']:.2f} V/s, V={s['value']:.3f}V"
            )

    print("\n".join(lines))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
