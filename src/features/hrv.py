"""
Heart Rate Variability (HRV) Feature Extraction Module
Course: Máy học nâng cao (Advanced Machine Learning)
"""

import numpy as np

def compute_hrv_features(hr_sequence, baseline_hr=80.0):
    """
    Extracts time-domain statistical features from a heart rate (HR) sequence.
    hr_sequence: array-like of shape (T,) in beats per minute (BPM)
    baseline_hr: resting/baseline driving heart rate
    
    Returns:
        dict containing statistical HRV metrics:
        - mean_hr
        - std_hr (approx. SDNN)
        - rmssd
        - max_hr
        - min_hr
        - delta_hr_pct
        - pnn50_approx
    """
    hr_arr = np.array(hr_sequence, dtype=np.float64)
    if len(hr_arr) < 2:
        return {
            "mean_hr": float(hr_arr[0]) if len(hr_arr) > 0 else baseline_hr,
            "std_hr": 0.0,
            "rmssd": 0.0,
            "max_hr": float(hr_arr[0]) if len(hr_arr) > 0 else baseline_hr,
            "min_hr": float(hr_arr[0]) if len(hr_arr) > 0 else baseline_hr,
            "delta_hr_pct": 0.0,
            "pnn50_approx": 0.0
        }

    # Inter-beat intervals (RR intervals in milliseconds approx = 60000 / HR)
    rr_intervals = 60000.0 / np.clip(hr_arr, 30.0, 220.0)
    diff_rr = np.diff(rr_intervals)

    mean_hr = float(np.mean(hr_arr))
    std_hr = float(np.std(hr_arr, ddof=1)) if len(hr_arr) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    max_hr = float(np.max(hr_arr))
    min_hr = float(np.min(hr_arr))
    latest_hr = float(hr_arr[-1])
    delta_hr_pct = float((latest_hr - baseline_hr) / baseline_hr * 100.0)
    pnn50_approx = float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0)

    return {
        "mean_hr": round(mean_hr, 2),
        "std_hr": round(std_hr, 2),
        "rmssd": round(rmssd, 2),
        "max_hr": round(max_hr, 2),
        "min_hr": round(min_hr, 2),
        "delta_hr_pct": round(delta_hr_pct, 2),
        "pnn50_approx": round(pnn50_approx, 2)
    }
