<<<<<<< HEAD
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def find_wheel_well_y(frame, x_center, y_start, search_height=150, threshold=180):
    """Scans vertically upwards from the top of the wheel to find the transition

    from a dark gap to the white car body (the wheel well line).
    """
    frame_height, frame_width = frame.shape[:2]

    # Constrain coordinates to frame boundaries
    x_center = max(0, min(frame_width - 1, x_center))
    y_start = max(0, min(frame_height - 1, y_start))
    y_end = max(0, y_start - search_height)

    # If the wheel is at the very top, return 0
    if y_start <= 0 or y_start == y_end:
        return y_end

    # Extract a 1-pixel wide vertical strip using a range [x:x+1]
    strip = frame[y_end:y_start, x_center : x_center + 1]

    # Double-check that the slice actually contains pixel data
    if strip.size == 0:
        return y_end

    # Convert to grayscale to evaluate brightness safely
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Flatten and flip the strip so we scan from the wheel upwards
    gray_strip = np.flip(gray_strip.flatten())

    for i, brightness in enumerate(gray_strip):
        # If the pixel brightness crosses into the "white car body" zone (0-255 scale)
        if brightness >= threshold:
            # Calculate the global Y coordinate in the frame
            return y_start - i

    # Fallback if no clean white transition is found
    return y_end


def process_video_wheels_only(
    video_path,
    output_video_path="wheels_tracked.mp4",
    excel_path="wheel_telemetry.xlsx",
):
    # 1. Load the wheel model
    wheel_model = YOLO("yolov11nWheel.pt")

    # 2. Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps, (frame_width, frame_height)
    )

    telemetry_data = []
    frame_count = 0

    # Dictionary to dynamically map the persistent tracking IDs (e.g., track_id 1, 2, 3...)
    # to human-readable names based on which one appeared first ("Wheel_1", "Wheel_2")
    wheel_id_mapping = {}

    print("Processing video frames and tracking wheels...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- FIX: Use wheel_model.track instead of wheel_model() ---
        # persist=True maintains identity across frames.
        wheel_results = wheel_model.track(
            frame, persist=True, verbose=False, conf=0.77
        )[0]

        # Ensure boxes and track IDs exist in the current frame
        if (
            wheel_results.boxes is not None
            and wheel_results.boxes.id is not None
        ):
            boxes = wheel_results.boxes.xyxy.cpu().numpy().astype(int)
            confs = wheel_results.boxes.conf.cpu().numpy()
            track_ids = wheel_results.boxes.id.cpu().numpy().astype(int)

            for box, conf, track_id in zip(boxes, confs, track_ids):
                wx1, wy1, wx2, wy2 = box

                # Dynamically register tracking IDs to permanent labels
                if track_id not in wheel_id_mapping:
                    assigned_index = len(wheel_id_mapping) + 1
                    wheel_id_mapping[track_id] = f"Wheel_{assigned_index}"

                wheel_id = wheel_id_mapping[track_id]

                # Calculate mid-point X of the wheel to scan straight upwards
                w_center_x = int((wx1 + wx2) / 2)

                # Find the wheel well Y coordinate using pixel intensity
                wheel_well_y = find_wheel_well_y(frame, w_center_x, wy1)

                # Calculate pixel gap distance
                clearance_px = wy1 - wheel_well_y

                # Append to dataset
                telemetry_data.append(
                    {
                        "Frame": frame_count,
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

                # --- Visual Render Overlay ---
                # Green Box for Wheel
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)

                # Blue Line representing the detected Wheel Well height line
                cv2.line(
                    frame,
                    (wx1, wheel_well_y),
                    (wx2, wheel_well_y),
                    (255, 0, 0),
                    2,
                )

                # Yellow Line indicating the measured distance vertical gap
                cv2.line(
                    frame,
                    (w_center_x, wy1),
                    (w_center_x, wheel_well_y),
                    (0, 255, 255),
                    2,
                )

                # HUD text labels
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

    # 5. Compile and Export Telemetry Data to Excel
    if telemetry_data:
        print("Compiling data into Excel sheet...")
        df = pd.DataFrame(telemetry_data)
        df.to_excel(excel_path, index=False)
        print(f"Data saved successfully! Generated: '{excel_path}'")
    else:
        print(
            "Warning: No wheels were detected in the video stream. Excel not generated."
        )


if __name__ == "__main__":
    process_video_wheels_only("undistorted_video.mp4")
=======
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def find_wheel_well_y(frame, x_center, y_start, search_height=150, threshold=180):
    """Scans vertically upwards from the top of the wheel to find the transition

    from a dark gap to the white car body (the wheel well line).
    """
    frame_height, frame_width = frame.shape[:2]

    # Constrain coordinates to frame boundaries
    x_center = max(0, min(frame_width - 1, x_center))
    y_start = max(0, min(frame_height - 1, y_start))
    y_end = max(0, y_start - search_height)

    # If the wheel is at the very top, return 0
    if y_start <= 0 or y_start == y_end:
        return y_end

    # Extract a 1-pixel wide vertical strip using a range [x:x+1]
    strip = frame[y_end:y_start, x_center : x_center + 1]

    # Double-check that the slice actually contains pixel data
    if strip.size == 0:
        return y_end

    # Convert to grayscale to evaluate brightness safely
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Flatten and flip the strip so we scan from the wheel upwards
    gray_strip = np.flip(gray_strip.flatten())

    for i, brightness in enumerate(gray_strip):
        # If the pixel brightness crosses into the "white car body" zone (0-255 scale)
        if brightness >= threshold:
            # Calculate the global Y coordinate in the frame
            return y_start - i

    # Fallback if no clean white transition is found
    return y_end


def process_video_wheels_only(
    video_path,
    output_video_path="wheels_tracked.mp4",
    excel_path="wheel_telemetry.xlsx",
):
    # 1. Load the wheel model
    wheel_model = YOLO("yolov11nWheel.pt")

    # 2. Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps, (frame_width, frame_height)
    )

    telemetry_data = []
    frame_count = 0

    # Dictionary to dynamically map the persistent tracking IDs (e.g., track_id 1, 2, 3...)
    # to human-readable names based on which one appeared first ("Wheel_1", "Wheel_2")
    wheel_id_mapping = {}

    print("Processing video frames and tracking wheels...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- FIX: Use wheel_model.track instead of wheel_model() ---
        # persist=True maintains identity across frames.
        wheel_results = wheel_model.track(
            frame, persist=True, verbose=False, conf=0.77
        )[0]

        # Ensure boxes and track IDs exist in the current frame
        if (
            wheel_results.boxes is not None
            and wheel_results.boxes.id is not None
        ):
            boxes = wheel_results.boxes.xyxy.cpu().numpy().astype(int)
            confs = wheel_results.boxes.conf.cpu().numpy()
            track_ids = wheel_results.boxes.id.cpu().numpy().astype(int)

            for box, conf, track_id in zip(boxes, confs, track_ids):
                wx1, wy1, wx2, wy2 = box

                # Dynamically register tracking IDs to permanent labels
                if track_id not in wheel_id_mapping:
                    assigned_index = len(wheel_id_mapping) + 1
                    wheel_id_mapping[track_id] = f"Wheel_{assigned_index}"

                wheel_id = wheel_id_mapping[track_id]

                # Calculate mid-point X of the wheel to scan straight upwards
                w_center_x = int((wx1 + wx2) / 2)

                # Find the wheel well Y coordinate using pixel intensity
                wheel_well_y = find_wheel_well_y(frame, w_center_x, wy1)

                # Calculate pixel gap distance
                clearance_px = wy1 - wheel_well_y

                # Append to dataset
                telemetry_data.append(
                    {
                        "Frame": frame_count,
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

                # --- Visual Render Overlay ---
                # Green Box for Wheel
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)

                # Blue Line representing the detected Wheel Well height line
                cv2.line(
                    frame,
                    (wx1, wheel_well_y),
                    (wx2, wheel_well_y),
                    (255, 0, 0),
                    2,
                )

                # Yellow Line indicating the measured distance vertical gap
                cv2.line(
                    frame,
                    (w_center_x, wy1),
                    (w_center_x, wheel_well_y),
                    (0, 255, 255),
                    2,
                )

                # HUD text labels
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

    # 5. Compile and Export Telemetry Data to Excel
    if telemetry_data:
        print("Compiling data into Excel sheet...")
        df = pd.DataFrame(telemetry_data)
        df.to_excel(excel_path, index=False)
        print(f"Data saved successfully! Generated: '{excel_path}'")
    else:
        print(
            "Warning: No wheels were detected in the video stream. Excel not generated."
        )


if __name__ == "__main__":
    process_video_wheels_only("undistorted_video.mp4")
>>>>>>> e6f7b7b14c47e3270059af06c2f5225015cef6c2
