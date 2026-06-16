"""
diagnose_artifacts.py
Run this against the SAME folder you load in Calcify to see what the
auto motion-artifact detector is actually doing: the global signal, the
z-score, the threshold, and which regions it flags.

Usage:
    python diagnose_artifacts.py
(edit FOLDER below to point at your plane0 directory)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter

# --- Configuration (edit before running) ---
FOLDER = "/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/Calcium Imaging/Nodose/Data/4_15_26/TSeries-04152026-1107-DCZ_REAL-003/suite2p/plane0"
# Detection parameters; keep these in sync with main.py for comparable output.
CUTOFF_HZ = 0.05
Z_THRESH  = 1.5
MIN_LEN   = 1
USE_STD   = False     # False = population mean, True = population std
# -------------------------------------------


def highpass_filter(signal, fs, cutoff_hz, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="high", analog=False)
    return filtfilt(b, a, signal)


def mask_to_regions(mask, min_len=1):
    regions, inr, start = [], False, 0
    for i, v in enumerate(mask):
        if v and not inr:
            start, inr = i, True
        elif not v and inr:
            end = i - 1
            if end - start + 1 >= min_len:
                regions.append((start, end))
            inr = False
    if inr:
        end = len(mask) - 1
        if end - start + 1 >= min_len:
            regions.append((start, end))
    return regions


def load_dff(folder):
    F = np.load(os.path.join(folder, "F.npy"))
    ops = np.load(os.path.join(folder, "ops.npy"), allow_pickle=True).item()
    fs = float(ops["fs"])
    # Apply the iscell filter (matches the Calcify default)
    iscell_path = os.path.join(folder, "iscell.npy")
    if os.path.exists(iscell_path):
        mask = np.load(iscell_path)[:, 0].astype(bool)
        F = F[mask]
    df = pd.DataFrame(F)
    # ΔF/F
    baseline = np.nanmean(df.values, axis=1, keepdims=True)
    baseline[baseline == 0] = np.nan
    dff = (df.values - baseline) / baseline
    dff = np.nan_to_num(dff)
    return pd.DataFrame(dff), fs


def main():
    df, fs = load_dff(FOLDER)
    print(f"fs={fs:.4f}  ROIs={df.shape[0]}  samples={df.shape[1]}  "
          f"duration={df.shape[1]/fs:.1f}s")

    global_signal = df.std(axis=0).values if USE_STD else df.mean(axis=0).values
    filtered = highpass_filter(global_signal, fs=fs, cutoff_hz=CUTOFF_HZ)

    mu, sigma = np.median(filtered), np.std(filtered)
    z = (filtered - mu) / sigma
    mask = np.abs(z) >= Z_THRESH
    regions = mask_to_regions(mask, min_len=MIN_LEN)

    print(f"\nDetected {len(regions)} regions at z>={Z_THRESH}:")
    for s, e in regions:
        print(f"  samples {s:5d}-{e:5d}  ({s/fs:7.1f}s - {e/fs:7.1f}s, "
              f"len {e-s+1})")

    t = np.arange(df.shape[1]) / fs
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    # Raw global signal with a few example ROIs
    for i in range(min(8, df.shape[0])):
        ax[0].plot(t, df.iloc[i], lw=0.5, alpha=0.4)
    ax[0].plot(t, global_signal, "k", lw=1.2, label="global mean")
    ax[0].set_title("ΔF/F traces (thin) + global signal (black)")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, filtered, lw=0.8)
    ax[1].set_title(f"High-pass filtered global signal (cutoff {CUTOFF_HZ} Hz)")

    ax[2].plot(t, z, lw=0.8)
    ax[2].axhline(Z_THRESH, color="r", ls="--", lw=0.8)
    ax[2].axhline(-Z_THRESH, color="r", ls="--", lw=0.8)
    ax[2].set_title(f"Robust z-score (threshold ±{Z_THRESH})")
    ax[2].set_xlabel("Time (s)")

    for s, e in regions:
        for a in ax:
            a.axvspan(s / fs, e / fs, color="red", alpha=0.25)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
