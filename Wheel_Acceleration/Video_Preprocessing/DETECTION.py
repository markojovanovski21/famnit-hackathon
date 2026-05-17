import glob
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def find_wheel_well_y(frame, x_center, y_start, search_height=150, threshold=180):
    """Scans vertically upwards from the top of the wheel to find the transition
    from a dark gap to the white car body (the wheel well line).
    """
    frame_height, frame_width = frame.shape[:2]

    x_center = max(0, min(frame_width - 1, x_center))
    y_start = max(0, min(frame_height - 1, y_start))
    y_end = max(0, y_start - search_height)

    if y_start <= 0 or y_start == y_end:
        return y_end

    # Slicing a 1-pixel wide column
    strip = frame[y_end:y_start, x_center, :]

    if strip.size == 0:
        return y_end

    # Fast grayscale conversion for a single column vector
    gray_strip = cv2.cvtColor(strip[..., np.newaxis, :], cv2.COLOR_BGR2GRAY)[
        :, 0
    ]

    # Search from bottom (y_start) to top (y_end)
    # Reversing via slicing [::-1] avoids deep memory copies
    for i, brightness in enumerate(gray_strip[::-1]):
        if brightness >= threshold:
            return y_start - i

    return y_end


def process_compressed_video(video_path, output_video_path, csv_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read original FPS from metadata
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps < 1.0 or np.isnan(original_fps):
        original_fps = 240.0  # Fallback

    target_playback_fps = 60.0
    frame_skip_interval = max(1, round(original_fps / target_playback_fps))

    print(f"Original Video FPS: {original_fps:.2f}")
    print(
        f"Processing sequentially for H.265 stability (Evaluating every {frame_skip_interval}th frame)"
    )

    # Output using XVID codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        target_playback_fps,
        (frame_width, frame_height),
    )

    telemetry_data = []
    raw_frame_count = 0
    saved_frame_count = 0
    wheel_id_mapping = {}

    print(f"Processing frames for: {os.path.basename(video_path)}")

    while cap.isOpened():
        # Sequential read keeps the H.265 decoder happy and moving forward smoothly
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame_count += 1

        # LIGHTWEIGHT SKIP: Skip heavy processing immediately without breaking the decode stream
        if raw_frame_count % frame_skip_interval != 0:
            continue

        saved_frame_count += 1

        # Run YOLO inference ONLY on our sample frames
        wheel_results = model.track(
            frame, persist=True, verbose=False, conf=0.77
        )[0]

        if (
            wheel_results.boxes is not None
            and wheel_results.boxes.id is not None
        ):
            boxes = wheel_results.boxes.xyxy.cpu().numpy().astype(int)
            confs = wheel_results.boxes.conf.cpu().numpy()
            track_ids = wheel_results.boxes.id.cpu().numpy().astype(int)

            for box, conf, track_id in zip(boxes, confs, track_ids):
                wx1, wy1, wx2, wy2 = box

                if track_id not in wheel_id_mapping:
                    assigned_index = len(wheel_id_mapping) + 1
                    wheel_id_mapping[track_id] = f"Wheel_{assigned_index}"

                wheel_id = wheel_id_mapping[track_id]
                w_center_x = int((wx1 + wx2) / 2)
                wheel_well_y = find_wheel_well_y(frame, w_center_x, wy1)
                clearance_px = wy1 - wheel_well_y

                telemetry_data.append(
                    {
                        "Original_Video_Frame": raw_frame_count,
                        "Output_Video_Frame": saved_frame_count,
                        "Wheel_ID": wheel_id,
                        "Confidence": round(float(conf), 2),
                        "Wheel_X1": wx1,
                        "Wheel_Y1": wy1,
                        "Wheel_X2": wx2,
                        "Wheel_Y2": wy2,
                        "Wheel_Well_Y": wheel_well_y,
                        "Clearance_Distance_px": clearance_px,
                    }
                )

                # --- Visual Render Overlays ---
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)
                cv2.line(
                    frame,
                    (wx1, wheel_well_y),
                    (wx2, wheel_well_y),
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    frame,
                    (w_center_x, wy1),
                    (w_center_x, wheel_well_y),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{wheel_id} Gap: {clearance_px}px",
                    (wx1, wy1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        out.write(frame)

    cap.release()
    out.release()
    print("Video encoding finished.")

    # Compile and Export Telemetry Data to CSV
    if telemetry_data:
        print("Compiling data into CSV file...")
        df = pd.DataFrame(telemetry_data)
        df.to_csv(csv_path, index=False)
        print(f"Data saved successfully! Generated: '{csv_path}'")
    else:
        print(
            "Warning: No wheels were detected in the video stream. CSV not generated."
        )


if __name__ == "__main__":
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []

    print("Scanning directory for video files...")
    for ext in video_extensions:
        video_files.extend(glob.glob(ext))

    if not video_files:
        print(f"No video files found in the directory: {os.getcwd()}")
    else:
        print(f"Found {len(video_files)} video(s) to process.")

        # Load the AI model ONCE outside the loop to save RAM/CPU initialization overhead
        print("Loading YOLO Wheel Model...")
        wheel_model = YOLO("yolov11nWheel.pt")

        for video in video_files:
            video_name, _ = os.path.splitext(video)

            # Avoid re-processing output directories if the script runs multiple times
            output_dir = f"{video_name}_output"
            os.makedirs(output_dir, exist_ok=True)

            output_video = os.path.join(output_dir, f"{video_name}_tracked.avi")
            output_csv = os.path.join(
                output_dir, f"{video_name}_telemetry.csv"
            )

            print(f"\n--- Starting processing for: {video} ---")
            process_compressed_video(
                video, output_video, output_csv, wheel_model
            )
            print(
                f"Finished processing for: {video}. Outputs saved in folder: '{output_dir}/'"
            )