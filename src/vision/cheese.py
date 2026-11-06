import subprocess
import numpy as np
import cv2
import os

# 输出目录
save_dir = "./calibration_images_3"
os.makedirs(save_dir, exist_ok=True)

# 摄像头参数
w, h = 1920, 1080
frame_size = w * h * 3

ffmpeg_cmd = [
    "ffmpeg",
    "-f", "v4l2",
    "-rtbufsize", "8M",
    "-input_format", "mjpeg",
    "-flags", "low_delay",
    "-framerate", "30",
    "-video_size", f"{w}x{h}",
    "-i", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_5630B8AF-video-index0",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an", "-sn", "-f", "rawvideo", "-"
]

pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=w*h*3)

img_id = 13

print("启动成功：按 <空格> 拍照，ESC 退出。")

while True:
    raw_frame = pipe.stdout.read(frame_size * 2)
    if not raw_frame:
        continue

    if len(raw_frame) >= frame_size:
        raw_frame = raw_frame[-frame_size:]
        try:
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3)).copy()
        except ValueError:
            continue

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # 空格
            img_id += 1
            filename = os.path.join(save_dir, f"img_{img_id:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"保存 {filename}")

pipe.stdout.close()
pipe.terminate()
pipe.wait()
cv2.destroyAllWindows()
