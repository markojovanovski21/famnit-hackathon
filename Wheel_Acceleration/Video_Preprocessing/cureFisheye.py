from datetime import datetime
import glob
import os
import cv2
import numpy as np


def batch_undistort_videos(search_pattern="*.mp4"):
    # Target both lowercase and uppercase variants to catch everything (.mp4 and .MP4)
    if search_pattern.endswith(".mp4"):
        alt_pattern = search_pattern[:-4] + ".MP4"
    elif search_pattern.endswith(".MP4"):
        alt_pattern = search_pattern[:-4] + ".mp4"
    else:
        alt_pattern = search_pattern

    # Gather files from both patterns
    video_files = glob.glob(search_pattern) + glob.glob(alt_pattern)
    
    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))

    # Filter out files that were already undistorted by this script to prevent infinite feedback loops
    video_files = [f for f in video_files if not os.path.basename(f).startswith("undistorted_")]

    if not video_files:
        print(f"No valid MP4 video files found matching the pattern.")
        return

    print(f"Found {len(video_files)} video(s) to process: {video_files}\n")

    for video_path in video_files:
        base_name = os.path.basename(video_path)
        output_name = f"undistorted_{base_name}"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing: {base_name}")
        print(f"--> Output target: {output_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}. Skipping.")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Fallback if FPS metadata is missing or corrupted
        if fps < 1.0 or np.isnan(fps):
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        # --- Custom Camera & Lens Parameters ---
        focal_length = width * 0.8
        K = np.array(
            [
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Stronger barrel correction configuration matrix
        D = np.array([-0.5, 0.2, 0, 0], dtype=np.float32)

        user_quit = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply geometric transformation matrix math
                undistorted = cv2.undistort(frame, K, D)
                out.write(undistorted)

                # Show real-time frame progress feedback
                cv2.imshow("Batch Undistort Preview", undistorted)

                # Press 'q' to stop processing completely
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nProcessing interrupted by user execution.")
                    user_quit = True
                    break
        finally:
            # Clean resources safely for this iteration loop
            cap.release()
            out.release()

        if user_quit:
            break

        print(f"Finished exporting: {output_name}\n")

    cv2.destroyAllWindows()
    print("All batch processing pipelines completed successfully.")


if __name__ == "__main__":
    # This will now automatically search for and process all .mp4 and .MP4 files in the directory
    batch_undistort_videos()