"""gta6-twitch-ccu-forecast-01: an analog-shape ensemble forecaster (Python 3.13, stdlib only, deterministic).

Methodology:

1. Analog shape library -- for each comparable launch, take its hourly viewers
   series and build the release-day ratio curve
   r_h = max(viewers_h, 1) / max(viewers_1, 1) for h = 2..24. Missing hours are
   interpolated in log space across the hour offsets; ratios are clamped to
   [0.05, 6.0]. Comparables with fewer than 8 hourly points on release day or
   with hour-1 viewers below 50 are rejected as too noisy to normalize.

2. Peak alignment -- rolling-midnight-local releases peak mid-day / evening UTC
   when anchored at their own 00:00Z hour, so anchoring every ratio at hour 1
   misaligns peaks. For each comparable, the release-day argmax offset is found;
   if it lies outside hours 1..6 (offsets 0..5), the comparable's hourly series
   is rolled so its peak lands at the case's implied unlock hour (derived from
   `notes`: simultaneous 00:00Z -> offset 0, 04:00Z / Eastern or rolling
   midnight-local -> offset 4), and ratios are recomputed on the rolled series,
   re-anchored at the aligned hour-1 value. Each comparable's contribution to the
   median is weighted by exp(-|peak_offset - unlock_offset| / 6), so badly
   misaligned launches barely count.

3. Analog aggregate -- for each hour h, the log1p(hour-1-channels)-weighted median
   of the peak-aligned r_h across comparables, plus the 5th/25th/75th/95th
   percentiles of the ratio distribution and the count of contributing
   comparables.

4. Naive persistence -- p_h = known_viewers * decay(h), where decay is a smooth
   exponential from 1 at hour 1 to the analog median's hour-24 ratio at hour 24,
   so persistence lands near the analog shape at the tail.

5. Shape sanity -- the blended center curve is enforced to decay gently and keep
   a visible EU-evening bump rather than collapsing near zero by hour 24:
   center(h) = max(center(h), known_viewers * 0.18) for every forecast hour by
   construction.

6. Buzz scaling -- a multiplicative tilt in log space, bounded to [0.7, 1.4],
   computed from the ratio of the case's pre-release-to-launch ratio
   (history_hourly baseline over the last 7 days and last 24 h before the
   checkpoint, against known hour-1 viewers) to the comparables' median same
   ratio. Too-sparse history ⇒ tilt = 1.

7. Ensemble -- log-space blend
   q50(h) = known * exp(w_analog * ln(median_r_h) + w_persist * ln(decay_h) + tilt)
   with w_analog = 0.75 / w_persist = 0.25 by default; w_analog is lowered toward
   0.5 where the comparables' interquartile log-spread at hour h is wide. If the
   hour-1 print is tiny (< 100 viewers) or no comparable covers hour h, the
   analog weight drops to 0 and the forecast falls back to pure persistence.

8. Quantile band -- q_tau(h) = center * exp(z_tau * sigma(h)),
   sigma(h) = sqrt(sigma_analog(h)^2 + sigma_model(h)^2),
   sigma_analog(h) = (ln p75 - ln p25) / 1.349,
   sigma_model(h) = 0.05 + 0.012 * h (pure persistence path uses sigma_model
   only), z = [-1.645, -0.674, 0, 0.674, 1.645]. q95 gets +15% spread for
   h <= 4 (launch spikes overshoot); q05 is floored at 1; each row is made
   non-decreasing and rounded half-up to 2 decimals (whole numbers as ints).
   Band targets: log-space sigma ~ 0.12-0.25 at hour 2, growing to ~ 0.35-0.55
   by hour 24.

9. Timezone/unlock handling -- the unlock style is derived from `notes`:
   rolling midnight-local / US Eastern 04:00Z unlocks get a ±0.10 log-space
   cosine wave (US 00:00-04:00Z damped, EU 17:00-22:00Z boosted); a global
   00:00Z unlock gets ±0.03. Fully data-derived from `known` + `comparables`;
   no game names are hardcoded.

10. Output -- every forecast emits the full hour 2..24 block (23 rows), filling
    in hours missing from the case's forecast rows. No network, no randomness,
    standard library only, deterministic output for identical input. Reads every
    <id>.in in MAKEFUN_INPUT_DIR (default /input), sorted by name, writes
    <id>.out to MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path: put it back first so a
# model.py shipped beside this file can be imported (kept for compatibility with the
# starter's invocation `python3 -P -s /bot/main.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RATIO_CAP = 6.0
RATIO_FLOOR = 0.05
MIN_COMP_POINTS = 8          # min hourly points a comparable needs on release day
MIN_COMP_HOUR1_VIEWERS = 50  # min hour-1 viewers to normalize a comparable

W_ANALOG = 0.75
W_PERSIST = 0.25
W_ANALOG_FLOOR = 0.5
SPREAD_CAP = 1.5  # log-units of IQR spread at which w_analog reaches its floor

TILT_FLOOR = 0.7
TILT_CAP = 1.4
SPARSE_MIN_POINTS = 3  # min history points needed to compute a tilt

SIGMA_MODEL_BASE = 0.05
SIGMA_MODEL_SLOPE = 0.012
Q95_EXTRA_HOURS = 4
Q95_EXTRA = 1.15

# Peak alignment. Rolling-midnight-local releases peak mid-day / evening UTC when
# anchored at their own 00:00Z hour, so anchoring every ratio at hour 1 misaligns
# peaks. For each comparable: find its release-day argmax offset; if it lies
# outside hours 1..6 (offsets 0..5), roll its hourly series so the peak lands at
# the case's implied unlock hour, then recompute ratios on the rolled series.
# Each comparable's contribution to the median is weighted by
# exp(-|peak_offset - unlock_offset| / ALIGN_TIME_CONSTANT), so badly misaligned
# launches barely count.
PEAK_EARLY_WINDOW = (0, 5)          # peak offsets (0-based) that need no alignment
UNLOCK_OFFSET_GLOBAL = 0            # simultaneous 00:00Z unlock
UNLOCK_OFFSET_ROLLING = 4           # rolling midnight local -> US Eastern 04:00Z anchor
ALIGN_TIME_CONSTANT = 6.0           # hours; alignment weight = exp(-|peak - unlock| / 6)

# Shape sanity. The blended median curve must decay gently and keep a visible
# EU-evening bump rather than collapsing near zero by hour 24: enforce
# center(h) >= known_viewers * TAIL_FLOOR_RATIO for every forecast hour by
# construction.
TAIL_FLOOR_RATIO = 0.18

# Edge hardening: an hour-1 print below this many viewers is too noisy to anchor
# analog shapes; such cases fall back to pure persistence.
TINY_KNOWN_VIEWERS = 100.0


def round2(x: float):
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def percentile_sorted(sorted_vals, p: float) -> float:
    """Linear-interpolation percentile (numpy default) on an ascending sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def weighted_median(vals, weights):
    """Weighted median: first value whose cumulative weight reaches half the total."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    total = sum(weights)
    if total <= 0:
        return percentile_sorted(sorted(vals), 50)
    cum = 0.0
    half = total / 2.0
    for i in order:
        cum += weights[i]
        if cum >= half:
            return vals[i]
    return vals[order[-1]]


def interpolate_log_series(points, target_offset: float) -> float:
    """Interpolate viewers (log space, linear in hour offset) at target_offset.

    points: list of (offset_hours, viewers). Values are clamped to >= 1 before
    logging; outside the observed range the nearest endpoint is used (constant in
    log space).
    """
    pts = sorted(((off, max(v, 1.0)) for off, v in points), key=lambda t: t[0])
    if not pts:
        return 1.0
    if target_offset <= pts[0][0]:
        return pts[0][1]
    if target_offset >= pts[-1][0]:
        return pts[-1][1]
    # bracket the target
    for i in range(len(pts) - 1):
        o0, v0 = pts[i]
        o1, v1 = pts[i + 1]
        if o0 <= target_offset <= o1:
            if o1 == o0:
                return v0
            t = (target_offset - o0) / (o1 - o0)
            return math.exp(math.log(v0) * (1.0 - t) + math.log(v1) * t)
    return pts[-1][1]


def comparable_ratio_curve(comp: dict, unlock_offset: int):
    """Return (r_h dict h=2..24, viewers_1, weight, n_points, peak_offset) or None.

    viewers_1 is the comparable's own release-day hour-1 value (used only for the
    noise filter). Ratios are computed on the peak-aligned series: if the
    comparable's release-day peak falls outside hours 1..6 (offsets 0..5), its
    hourly series is rolled so the peak lands at unlock_offset, and ratios are
    recomputed re-anchored at the aligned hour-1 value.
    """
    day_start = parse_ts(comp["day_start"])
    hourly = comp.get("hourly") or []
    # release-day points: [day_start, day_start + 24 h)
    pts = []
    for x in hourly:
        t = parse_ts(x["from"])
        if day_start <= t < day_start + timedelta(hours=24):
            offset = (t - day_start).total_seconds() / 3600.0
            pts.append((offset, float(x.get("viewers", 0) or 0)))
    if len(pts) < MIN_COMP_POINTS:
        return None
    viewers_1 = interpolate_log_series(pts, 0.0)
    if viewers_1 < MIN_COMP_HOUR1_VIEWERS:
        return None
    # release-day argmax offset (first maximum on ties)
    peak_offset = min(pts, key=lambda t: (-t[1], t[0]))[0]

    shift = 0
    if not (PEAK_EARLY_WINDOW[0] <= peak_offset <= PEAK_EARLY_WINDOW[1]):
        shift = (unlock_offset - int(round(peak_offset))) % 24
        pts = [((off + shift) % 24.0, v) for off, v in pts]

    anchor = max(interpolate_log_series(pts, 0.0), 1.0)
    ratios = {}
    for h in range(2, 25):
        v = interpolate_log_series(pts, h - 1)
        r = max(v, 1.0) / anchor
        ratios[h] = min(RATIO_CAP, max(RATIO_FLOOR, r))
    # weight from hour-1 channels of the comparable (exact point at offset 0 if present)
    hour1_channels = 0.0
    for x in hourly:
        t = parse_ts(x["from"])
        if t == day_start:
            hour1_channels = float(x.get("channels", 0) or 0)
            break
    weight = math.log1p(hour1_channels)
    # down-weight launches whose peak sits far from the case's unlock hour
    weight *= math.exp(-abs(peak_offset - unlock_offset) / ALIGN_TIME_CONSTANT)
    return ratios, viewers_1, weight, len(pts), peak_offset, shift


def analog_library(comparables: list, unlock_offset: int):
    """Build per-hour ratio distributions across valid comparables.

    Each comparable's ratio curve is peak-aligned to unlock_offset and its
    contribution to the median is weighted by exp(-|peak - unlock| / 6), so
    badly misaligned launches barely count.

    Returns dict hour -> {median, p5, p25, p75, p95, count} plus hour-24 median.
    """
    per_hour = {h: [] for h in range(2, 25)}
    for comp in comparables:
        cur = comparable_ratio_curve(comp, unlock_offset)
        if cur is None:
            continue
        ratios, _v1, weight, _n, _peak, _shift = cur
        for h in range(2, 25):
            per_hour[h].append((ratios[h], weight))
    agg = {}
    for h in range(2, 25):
        items = per_hour[h]
        if not items:
            agg[h] = {"median": 1.0, "p5": 1.0, "p25": 1.0, "p75": 1.0, "p95": 1.0, "count": 0}
            continue
        vals = [r for r, _ in items]
        wts = [w for _, w in items]
        svals = sorted(vals)
        agg[h] = {
            "median": weighted_median(vals, wts),
            "p5": percentile_sorted(svals, 5),
            "p25": percentile_sorted(svals, 25),
            "p75": percentile_sorted(svals, 75),
            "p95": percentile_sorted(svals, 95),
            "count": len(vals),
        }
    return agg


def pre_release_baseline(hourly, before: datetime, window_hours: int):
    """Mean viewers of points in [before - window, before)."""
    start = before - timedelta(hours=window_hours)
    vals = []
    for x in hourly:
        t = parse_ts(x["from"])
        if start <= t < before:
            vals.append(float(x.get("viewers", 0) or 0))
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def implied_unlock_offset(notes: str) -> int:
    """Case's implied unlock hour as a UTC offset (0-based; hour 1 == offset 0).

    Derived from `notes` only: a simultaneous 00:00Z unlock is offset 0, an
    explicit 04:00Z / Eastern unlock is offset 4, and a rolling midnight-local
    release defaults to the US Eastern 04:00Z anchor. No game names are hardcoded.
    """
    n = (notes or "").lower()
    if ("global" in n or "simultaneous" in n) and "00:00z" in n:
        return UNLOCK_OFFSET_GLOBAL
    if "04:00z" in n or "eastern" in n:
        return UNLOCK_OFFSET_ROLLING
    if "00:00z" in n:
        return UNLOCK_OFFSET_GLOBAL
    return UNLOCK_OFFSET_ROLLING


def tilt_factor(case: dict, known_viewers: float, comparables: dict, analog,
                unlock_offset: int) -> float:
    """Multiplicative buzz tilt in log space, clamped to [TILT_FLOOR, TILT_CAP]."""
    checkpoint = parse_ts(case["checkpoint"])
    hh = case.get("history_hourly") or []
    base24, n24 = pre_release_baseline(hh, checkpoint, 24)
    base7, n7 = pre_release_baseline(hh, checkpoint, 24 * 7)

    bases = []
    if base24 is not None and n24 >= SPARSE_MIN_POINTS:
        bases.append(base24)
    if base7 is not None and n7 >= SPARSE_MIN_POINTS:
        bases.append(base7)
    if not bases or known_viewers <= 0:
        return 1.0
    case_baseline = sum(bases) / len(bases)
    case_ratio = case_baseline / max(known_viewers, 1.0)

    # Comparables' median pre-release-to-launch ratio (last 7d + last 24h before day_start).
    comp_ratios = []
    for comp in comparables:
        day_start = parse_ts(comp["day_start"])
        hourly = comp.get("hourly") or []
        cb24, n24c = pre_release_baseline(hourly, day_start, 24)
        cb7, n7c = pre_release_baseline(hourly, day_start, 24 * 7)
        cbases = []
        if cb24 is not None and n24c >= SPARSE_MIN_POINTS:
            cbases.append(cb24)
        if cb7 is not None and n7c >= SPARSE_MIN_POINTS:
            cbases.append(cb7)
        if not cbases:
            continue
        comp_baseline = sum(cbases) / len(cbases)
        # anchor viewers_1 for this comparable
        cur = comparable_ratio_curve(comp, unlock_offset)
        if cur is None:
            continue
        _ratios, v1, _w, _n, _peak, _shift = cur
        if v1 <= 0:
            continue
        comp_ratios.append(comp_baseline / max(v1, 1.0))
    if not comp_ratios:
        return 1.0
    comp_median = percentile_sorted(sorted(comp_ratios), 50)
    if comp_median <= 0 or case_ratio <= 0:
        return 1.0
    tilt_log = math.log(case_ratio) - math.log(comp_median)
    tilt_log = min(math.log(TILT_CAP), max(math.log(TILT_FLOOR), tilt_log))
    return math.exp(tilt_log)


def unlock_amplitude(notes: str) -> float:
    """Derive unlock wave amplitude from the case notes (data-derived, no game names)."""
    n = (notes or "").lower()
    if "00:00z" in n and ("global" in n or "simultaneous" in n):
        return 0.03
    if "rolling" in n or "midnight local" in n or "04:00z" in n or "eastern" in n:
        return 0.10
    if "midnight" in n:
        return 0.10
    return 0.10


def unlock_wave_log(utc_hour: int, amplitude: float) -> float:
    """Log-space wave: US 00:00-04:00Z damped, EU 17:00-22:00Z boosted, cosine windows."""
    if 0 <= utc_hour < 4:
        t = utc_hour / 4.0
        bump = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))  # 0 -> 1 -> 0
        return -amplitude * bump
    if 17 <= utc_hour < 22:
        t = (utc_hour - 17) / 5.0
        bump = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
        return amplitude * bump
    return 0.0


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if int(k.get("hour", 0)) == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    known_viewers = max(1.0, float(first.get("viewers", 0) or 0))

    unlock_offset = implied_unlock_offset(case.get("notes") or "")
    analog = analog_library(case.get("comparables") or [], unlock_offset)
    tilt = tilt_factor(case, known_viewers, case.get("comparables") or [], analog,
                       unlock_offset)

    notes = case.get("notes") or ""
    amplitude = unlock_amplitude(notes)

    # Edge hardening: a tiny hour-1 print is too noisy to anchor analog shapes,
    # and an analog library with no coverage at hour h cannot contribute there;
    # both fall back to pure persistence with the fixed sigma schedule.
    use_analog = known_viewers >= TINY_KNOWN_VIEWERS

    # Always emit the full 2..24 block, filling in hours missing from the
    # case's forecast rows.
    hours_out = []
    for h in range(2, 25):
        utc_hour = h - 1  # hour 1 == 00:00-01:00 Z
        a = analog[h]
        med = a["median"]
        med24 = analog[24]["median"]
        decay = med24 ** ((h - 1) / 23.0)

        spread = math.log(max(a["p75"], 1e-9)) - math.log(max(a["p25"], 1e-9))
        w_analog = 0.0
        if use_analog and a["count"] > 0:
            w_analog = W_ANALOG - (W_ANALOG - W_ANALOG_FLOOR) * min(1.0, spread / SPREAD_CAP)
            w_analog = min(W_ANALOG, max(W_ANALOG_FLOOR, w_analog))
        w_persist = 1.0 - w_analog

        center_log = math.log(known_viewers)
        if use_analog and a["count"] > 0:
            center_log += w_analog * math.log(max(med, 1e-9))
        center_log += w_persist * math.log(max(decay, 1e-9))
        center_log += math.log(tilt)
        center_log += unlock_wave_log(utc_hour, amplitude)
        center = math.exp(center_log)

        # Shape sanity: the curve decays gently and keeps a visible EU-evening
        # bump; it never decays below a fixed fraction of hour-1 viewers.
        center = max(center, known_viewers * TAIL_FLOOR_RATIO)

        sigma_model = SIGMA_MODEL_BASE + SIGMA_MODEL_SLOPE * h
        if use_analog and a["count"] > 0:
            sigma_analog = (math.log(max(a["p75"], 1e-9))
                            - math.log(max(a["p25"], 1e-9))) / 1.349
            sigma = math.sqrt(sigma_analog * sigma_analog + sigma_model * sigma_model)
        else:
            sigma = sigma_model

        qs = []
        for z, tau in zip(Z, LEVELS):
            mult = Q95_EXTRA if (tau == 0.95 and h <= Q95_EXTRA_HOURS) else 1.0
            qs.append(center * math.exp(z * sigma * mult))

        # floor q05 at 1, then enforce non-decreasing within the row
        qs[0] = max(qs[0], 1.0)
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours_out.append({"hour": h, "q": [round2(q) for q in qs]})

    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
