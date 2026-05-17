import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def detect_oscillation_bounds(time, acceleration, fps, threshold_ratio=0.15):
    """Automatically detects when the acceleration starts and stops oscillating,

    trims 20% off both ends of that window, and returns the trimmed timestamps.
    """
    # Calculate a rolling absolute deviation to find where the "action" is
    window_size = int(fps * 0.2)  # 0.2 second rolling window
    if window_size < 3:
        window_size = 3

    rolling_abs = (
        pd.Series(np.abs(acceleration))
        .rolling(window=window_size, center=True)
        .mean()
        .fillna(0)
        .to_numpy()
    )

    # Threshold is set relative to the peak oscillation found
    threshold = np.max(rolling_abs) * threshold_ratio
    active_indices = np.where(rolling_abs > threshold)[0]

    if len(active_indices) < 2:
        # Fallback if no clear oscillation is detected
        return time[0], time[-1]

    # Find raw start and end of oscillation
    start_idx = active_indices[0]
    end_idx = active_indices[-1]

    oscillation_length = end_idx - start_idx

    # Trim 20% from the beginning and 20% from the end of the detected length
    trim_amount = int(oscillation_length * 0.20)
    trimmed_start_idx = start_idx + trim_amount
    trimmed_end_idx = end_idx - trim_amount

    # Ensure indices stay bounded correctly
    trimmed_start_idx = max(0, trimmed_start_idx)
    trimmed_end_idx = min(len(time) - 1, trimmed_end_idx)

    return time[trimmed_start_idx], time[trimmed_end_idx]


def process_csv_file(csv_path, fps=60):
    # Extract file name without extension for naming the graph
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_image_path = f"{base_name}.png"

    try:
        # Reading CSV with automatic separator detection (handles commas, semicolons, etc.)
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # Dynamic Column Resolution for Target Video Frames
    frame_col_opts = [
        col
        for col in df.columns
        if ("Output" in col and "Frame" in col) or "saved" in col
    ]
    if not frame_col_opts:
        frame_col_opts = [
            col for col in df.columns if "Video" in col or "Frame" in col
        ]

    frame_column_name = frame_col_opts[0] if frame_col_opts else "Frame"
    if frame_column_name not in df.columns:
        df[frame_column_name] = np.arange(1, len(df) + 1)

    # Resolution for clearance metrics
    dist_col = [
        col for col in df.columns if "Clearance" in col or "Distance" in col
    ]
    distance_column_name = (
        dist_col[0] if dist_col else "Clearance_Distance_px"
    )

    # Check for Wheel_ID column existence
    if "Wheel_ID" not in df.columns:
        print(f"Error: 'Wheel_ID' column missing in {csv_path}")
        return

    # Filter strictly for Wheel_1 and Wheel_2 if available, or fetch whatever exists
    unique_wheels = [
        w for w in ["Wheel_1", "Wheel_2"] if w in df["Wheel_ID"].unique()
    ]
    if not unique_wheels:
        unique_wheels = sorted(df["Wheel_ID"].unique())

    if not unique_wheels:
        print(f"No wheel data found in {csv_path}")
        return

    fig, axes = plt.subplots(
        len(unique_wheels), 1, figsize=(11, 4.5 * len(unique_wheels))
    )
    if len(unique_wheels) == 1:
        axes = [axes]

    for idx, technical_id in enumerate(unique_wheels):
        friendly_name = (
            "Front Wheel"
            if technical_id == "Wheel_1"
            else "Rear Wheel" if technical_id == "Wheel_2" else technical_id
        )
        line_color = "crimson" if technical_id == "Wheel_1" else "royalblue"

        # Isolate and sort telemetry elements sequentially
        wheel_df = df[df["Wheel_ID"] == technical_id].sort_values(
            by=frame_column_name
        )
        if len(wheel_df) < 10:
            continue

        dt = 1.0 / fps
        det_frames = wheel_df[frame_column_name].to_numpy()
        det_clearance = wheel_df[distance_column_name].to_numpy()
        det_time = det_frames / fps

        # Compute numerical derivatives
        velocity = np.diff(det_clearance) / dt
        acceleration = np.diff(velocity) / dt
        acceleration = np.pad(acceleration, (1, 1), mode="edge")

        # Savitzky-Golay filtering to clear sub-pixel vibration artifacts
        window_len = min(
            15, len(acceleration) - (1 if len(acceleration) % 2 == 0 else 0)
        )
        if window_len > 3:
            acceleration = savgol_filter(
                acceleration, window_length=window_len, polyorder=2
            )

        # Clear standard edge tracking initialization/termination noise spikes
        suppress_count = int(0.15 * fps)
        if len(acceleration) > (suppress_count * 2):
            acceleration[:suppress_count] = 0.0
            acceleration[-suppress_count:] = 0.0

        # --- AUTOMATION BLOCK: DETECT START AND END TIMES DYNAMICALLY ---
        plot_start_sec, plot_end_sec = detect_oscillation_bounds(
            det_time, acceleration, fps
        )

        # Apply the 1.5-second margin on either side
        padded_start_sec = max(det_time[0], plot_start_sec - 1.5)
        padded_end_sec = min(det_time[-1], plot_end_sec + 1.5)

        # Create the plot scale matching the padded time slice
        plot_start_frame = int(padded_start_sec * fps)
        plot_end_frame = int(padded_end_sec * fps)

        full_frame_range = np.arange(plot_start_frame, plot_end_frame + 1)
        full_time_range = full_frame_range / fps

        # Map analytical frame indices to the global plot space
        accel_map = dict(zip(det_frames, acceleration))
        final_acceleration = np.array(
            [accel_map.get(frame, 0.0) for frame in full_frame_range]
        )

        # Force a flat line outside the actual trimmed evaluation window
        outside_window_mask = (full_time_range < plot_start_sec) | (
            full_time_range > plot_end_sec
        )
        final_acceleration[outside_window_mask] = 0.0

        # Subplot Graph Generation
        ax = axes[idx]
        ax.plot(
            full_time_range,
            final_acceleration,
            label=f"{friendly_name} Pulse",
            color=line_color,
            linewidth=2.2,
        )

        # Highlight target calculation window
        ax.axvspan(
            plot_start_sec,
            plot_end_sec,
            color="gold",
            alpha=0.1,
            label="Trimmed Evaluation Zone (Middle 60%)",
        )

        # Typography and Grid adjustments
        ax.set_title(
            f"Vertical Acceleration Signature - {friendly_name} ({base_name})",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Playback Time (Seconds)", fontsize=10)
        ax.set_ylabel(r"Acceleration ($px/s^2$)", fontsize=10)

        # Lock axis boundaries strictly to the 1.5s padding window
        ax.set_xlim(padded_start_sec, padded_end_sec)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(
            loc="upper right", frameon=True, facecolor="white", framealpha=0.9
        )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated chart: '{output_image_path}' from {csv_path}")


if __name__ == "__main__":
    # Automatically finds every CSV file (*.csv) in your current folder
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files (.csv) found in the current folder.")
    else:
        print(f"Found {len(csv_files)} CSV file(s) to process.")
        for file in csv_files:
            print(f"Processing: {file}...")
            process_csv_file(csv_path=file, fps=60)
        print("All CSV files processed successfully!")