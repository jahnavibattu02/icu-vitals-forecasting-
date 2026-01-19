import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="synthetic_icu_vitals.csv")
    parser.add_argument("--patients", type=int, default=80)
    parser.add_argument("--timesteps", type=int, default=120)  # per patient
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for pid in range(args.patients):
        # Create smooth-ish signals with patient-specific baseline + random walks
        hr = 80 + rng.normal(0, 5)
        map_ = 75 + rng.normal(0, 4)
        spo2 = 96 + rng.normal(0, 1)

        for t in range(args.timesteps):
            # random walk + occasional drift
            hr += rng.normal(0, 1.2)
            map_ += rng.normal(0, 0.8)
            spo2 += rng.normal(0, 0.3)

            # clamp realistic-ish ranges
            hr_c = float(np.clip(hr, 40, 160))
            map_c = float(np.clip(map_, 45, 120))
            spo2_c = float(np.clip(spo2, 80, 100))

            # introduce some missingness
            if rng.random() < 0.03: hr_c = np.nan
            if rng.random() < 0.03: map_c = np.nan
            if rng.random() < 0.02: spo2_c = np.nan

            rows.append([pid, t, hr_c, map_c, spo2_c])

    df = pd.DataFrame(rows, columns=["patient_id", "timestamp", "HR", "MAP", "SpO2"])
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote synthetic dataset to {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
