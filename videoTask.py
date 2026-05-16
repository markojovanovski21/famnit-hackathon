import cv2
import pandas as pd
from ultralytics import YOLO


def process_video_with_telemetry(video_path, output_video_path="output_tracked.mp4", excel_path="detections_log.xlsx"):
    # 1. Load models
    car_model = YOLO("yolo11n.pt")
    wheel_model = YOLO("yolov11nWheel.pt")

    # 2. Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video output setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 3. Create an empty list to store row data for Excel
    telemetry_data = []
    
    frame_count = 0
    print("Processing video frames and calculating telemetry...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Step 1: Detect cars
        car_results = car_model(frame, verbose=False)[0]

        for box in car_results.boxes:
            if int(box.cls[0]) == 2:  # Class 2 is 'car'
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                car_conf = float(box.conf[0])
                
                car_width = cx2 - cx1
                car_height = cy2 - cy1

                # Append car metrics to our dataset list
                telemetry_data.append({
                    "Frame": frame_count,
                    "Object": "Car",
                    "Confidence": round(car_conf, 2),
                    "X1": cx1, "Y1": cy1, "X2": cx2, "Y2": cy2,
                    "Width_px": car_width,
                    "Height_px": car_height
                })

                # Draw Blue Bounding Box for Car
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 3)
                cv2.putText(frame, f"Car {car_conf:.2f}", (cx1, cy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Step 2: Crop Car and Detect Wheels
                pad = 10
                x1_pad, y1_pad = max(0, cx1 - pad), max(0, cy1 - pad)
                x2_pad, y2_pad = min(frame_width, cx2 + pad), min(frame_height, cy2 + pad)
                cropped_car = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                wheel_results = wheel_model(cropped_car, verbose=False)[0]

                for w_box in wheel_results.boxes:
                    wheel_conf = float(w_box.conf[0])

                    # ---- CONFIDENCE THRESHOLD FILTER ----
                    # Ignore any wheel detections below 55% (0.55) confidence
                    if wheel_conf < 0.77:
                        continue

                    wx1, wy1, wx2, wy2 = map(int, w_box.xyxy[0])

                    # Recalculate to global frame coordinates
                    global_wx1 = x1_pad + wx1
                    global_wy1 = y1_pad + wy1
                    global_wx2 = x1_pad + wx2
                    global_wy2 = y1_pad + wy2

                    wheel_width = global_wx2 - global_wx1
                    wheel_height = global_wy2 - global_wy1

                    # Append filtered wheel metrics to dataset list
                    telemetry_data.append({
                        "Frame": frame_count,
                        "Object": "Wheel",
                        "Confidence": round(wheel_conf, 2),
                        "X1": global_wx1, "Y1": global_wy1, "X2": global_wx2, "Y2": global_wy2,
                        "Width_px": wheel_width,
                        "Height_px": wheel_height
                    })

                    # Draw Green Bounding Box for Wheel
                    cv2.rectangle(frame, (global_wx1, global_wy1), (global_wx2, global_wy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Wheel {wheel_conf:.2f}", (global_wx1, global_wy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the visual frame
        out.write(frame)

    # 4. Cleanup media streams
    cap.release()
    out.release()
    print("Video encoding finished.")

    # 5. Convert compiled telemetry array to Pandas DataFrame & Export to Excel
    print("Compiling data into Excel sheet...")
    df = pd.DataFrame(telemetry_data)
    
    # Save to Excel (index=False prevents empty placeholder columns)
    df.to_excel(excel_path, index=False)
    print(f"Data saved successfully! Generated: '{excel_path}'")


if __name__ == "__main__":
    # Ensure you are targeting your 60fps video file here
    process_video_with_telemetry("EQE_01.mp4")