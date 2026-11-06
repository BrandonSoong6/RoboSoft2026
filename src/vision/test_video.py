import subprocess
import numpy as np
import cv2
import time

device_name = "HD Pro Webcam C920"
w, h = 1920, 1080

# ffmpeg 输出 BGR24 原始流
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "dshow",
    "-flags", "low_delay",
    "-framerate", "30",
    "-video_size", f"{w}x{h}",
    "-i", f"video={device_name}",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an", "-sn", "-f", "rawvideo", "-"
]

pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=w*h*3)

num_frames = 1200
print("开始采集...")

start = time.time()
for i in range(num_frames):
    raw_frame = pipe.stdout.read(w*h*3)
    if not raw_frame:
        break
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))

    cv2.imshow("C920 rawvideo", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
        break

end = time.time()
elapsed = end - start
fps = num_frames / elapsed
print(f"采集 {num_frames} 帧用时 {elapsed:.2f} 秒，实际 FPS ≈ {fps:.2f}")

pipe.terminate()
cv2.destroyAllWindows()
