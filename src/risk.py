from typing import Dict, List
import numpy as np

def trend_risk_score(
    window_last: np.ndarray,        # (T, F) raw
    pred_future: np.ndarray,        # (H, F) raw
    feature_names: List[str]
) -> Dict:
    """
    More sensitive + realistic demo risk score:
    - compares forecast to the last observed value
    - checks absolute thresholds (MAP/SpO2)
    - checks sustained downward trend
    """

    idx = {name: i for i, name in enumerate(feature_names)}
    map_i = idx.get("MAP")
    spo2_i = idx.get("SpO2")
    hr_i = idx.get("HR")

    if map_i is None or spo2_i is None:
        return {"risk_level": "Low", "score": 0, "reasons": ["Required features not found (MAP/SpO2)."]}

    # Last observed values (end of window)
    last_map = float(window_last[-1, map_i])
    last_spo2 = float(window_last[-1, spo2_i])
    last_hr = float(window_last[-1, hr_i]) if hr_i is not None else None

    map_pred = pred_future[:, map_i]
    spo2_pred = pred_future[:, spo2_i]
    hr_pred = pred_future[:, hr_i] if hr_i is not None else None

    min_map = float(np.min(map_pred))
    min_spo2 = float(np.min(spo2_pred))

    # drop relative to last observed (more meaningful)
    map_drop = last_map - min_map
    spo2_drop = last_spo2 - min_spo2

    # trend counts
    map_decreasing_steps = int(np.sum(np.diff(map_pred) < 0))
    spo2_decreasing_steps = int(np.sum(np.diff(spo2_pred) < 0))

    score = 0
    reasons = []

    # --- MAP rules ---
    # absolute threshold
    if min_map < 65:
        score += 2
        reasons.append(f"Forecasted MAP minimum is {min_map:.1f} (<65).")

    # relative drop + sustained trend (lowered thresholds for H=6 demo)
    if map_drop >= 4 and map_decreasing_steps >= 2:
        score += 2
        reasons.append(f"MAP shows a downward trend (drop ~{map_drop:.1f} from last observed).")

    # --- SpO2 rules ---
    if min_spo2 < 92:
        score += 2
        reasons.append(f"Forecasted SpO₂ minimum is {min_spo2:.1f}% (<92%).")

    if spo2_drop >= 1.5 and spo2_decreasing_steps >= 2:
        score += 2
        reasons.append(f"SpO₂ shows a downward trend (drop ~{spo2_drop:.1f}% from last observed).")

    # --- HR rule (optional mild) ---
    if hr_pred is not None and last_hr is not None:
        max_hr = float(np.max(hr_pred))
        hr_rise = max_hr - last_hr
        if hr_rise >= 10:
            score += 1
            reasons.append(f"Forecasted HR increases by ~{hr_rise:.1f} (from last observed).")

    # Map score -> level
    if score >= 6:
        level = "High"
    elif score >= 3:
        level = "Medium"
    else:
        level = "Low"

    if not reasons:
        reasons.append("No concerning MAP/SpO₂ thresholds or sustained downward trends detected.")

    return {
        "risk_level": level,
        "score": score,
        "reasons": reasons,
        "debug": {
            "last_map": last_map, "min_map": min_map, "map_drop": float(map_drop),
            "last_spo2": last_spo2, "min_spo2": min_spo2, "spo2_drop": float(spo2_drop),
            "map_decreasing_steps": map_decreasing_steps,
            "spo2_decreasing_steps": spo2_decreasing_steps,
        }
    }
