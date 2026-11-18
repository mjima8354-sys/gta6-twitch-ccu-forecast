# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mjima8354-sys/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mjima8354-sys/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour quantile forecast (q05/q25/q50/q75/q95, hours 2–24 UTC) of Twitch concurrent viewers in the Grand Theft Auto VI category on release day, built by an AI agent for make.fun.

## Methodology — analog-shape ensemble

`main.py` (Python 3.13, standard library only, fully deterministic, no network) forecasts each case from the data shipped inside it:

1. **Analog shape library** — every comparable launch's hourly category viewers are normalized to their own hour-1 print (`r_h = viewers_h / viewers_1`), missing hours interpolated in log space, ratios clamped to [0.05, 6.0]; noisy comparables (few points, tiny hour-1) are rejected.
2. **Peak alignment** — releases that unlock at rolling midnight-local peak mid-day UTC, so a comparable whose argmax falls outside the first 6 hours is rolled to align its peak with the case's implied unlock hour, and weighted by `exp(-|peak − unlock| / 6)`.
3. **Analog aggregate** — a channel-weighted median of the aligned ratios per hour plus the 5th/25th/75th/95th percentile dispersion.
4. **Naive persistence** — a smooth decay from hour 1's actual print to the analog median's hour-24 level.
5. **Log-space ensemble** — `q50(h) = known · exp(0.75·ln(median r_h) + 0.25·ln(decay) + tilt)`, where `tilt` (bounded [0.7, 1.4]) compares the case's pre-release buzz (history_hourly baseline vs. launch print) to the comparables' own, and the analog weight drops toward 0.5 where comparables disagree.
6. **Unlock waves** — a smooth cosine term (±0.10 log units for rolling/Eastern unlocks, ±0.03 for global 00:00Z) damps the US late-night hours and boosts the EU-evening band (17:00–22:00Z).
7. **Quantile band** — `q_τ = center · exp(z_τ · σ(h))` with σ(h) combining the comparables' robust log-ratio spread `(ln p75 − ln p25)/1.349` and a model term `0.05 + 0.012·h`, an asymmetric +15% upper-tail spike for the first 4 hours, a floor at hour 1 × 0.18, and monotone, half-up 2-decimal rounding.

Launch analogs behind the design: Cyberpunk 2077's 1.14M peak Twitch viewers at a midnight launch (Dec 2020), Elden Ring's 910K day-one peak with a mid-evening UTC crest (Feb 2022), and Baldur's Gate 3's 481.5K evening-unlock peak (Aug 2023) — the same pre-midnight spike / decay / timezone-wave shape the ensemble infers per case from its own comparables.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
