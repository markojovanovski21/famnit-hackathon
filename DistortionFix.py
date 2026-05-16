import cv2
import numpy as np

cap = cv2.VideoCapture("EQE_01.MP4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "undistorted_video.mp4",
    fourcc,
    fps,
    (width, height)
)
# Better approximate camera matrix
focal_length = width * 0.8

K = np.array([
    [focal_length, 0, width / 2],
    [0, focal_length, height / 2],
    [0, 0, 1]
], dtype=np.float32)

# Stronger barrel correction
# k1 negative removes barrel distortion
D = np.array([-0.5, 0.2, 0, 0], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    undistorted = cv2.undistort(frame, K, D)
    out.write(undistorted)
    cv2.imshow("Undistorted", undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()