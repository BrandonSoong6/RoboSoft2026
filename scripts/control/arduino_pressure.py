import serial
import threading
import time
import numpy as np
import struct

class ArduinoComm:
    def __init__(self,
                 port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                 baudrate=115200,
                 timeout=0.005): 
        self.ser = serial.Serial(
            port, baudrate=baudrate,
            timeout=timeout, write_timeout=0.5
        )
        time.sleep(2.0)  # 给 Arduino 复位时间
        self._stop_flag = False
        self.current_angles = (0.0, 0.0, 0.0)

        # 启动读取线程
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _reader(self):
        """后台线程：持续解析二进制帧 (0xAA 0x55 LEN data chk)"""
        buf = b""
        while not self._stop_flag:
            try:
                data = self.ser.read(128)
                if not data:
                    continue
                buf += data

                # 解析循环
                while True:
                    start = buf.find(b"\xAA\x55")
                    if start < 0:
                        buf = b""
                        break
                    if len(buf) < start + 3:
                        break
                    length = buf[start+2]
                    frame_len = 2 + 1 + length + 1
                    if len(buf) < start + frame_len:
                        break

                    frame = buf[start:start+frame_len]
                    buf = buf[start+frame_len:]

                    # 校验
                    chk_calc = 0
                    for b in frame[2:-1]:
                        chk_calc ^= b
                    if chk_calc != frame[-1]:
                        continue

                    data_bytes = frame[3:-1]
                    if len(data_bytes) % 2 != 0:
                        continue

                    vals_raw = struct.unpack("<" + "h"*(len(data_bytes)//2), data_bytes)
                    vals_deg = tuple(v/100.0 for v in vals_raw)
                    self.current_angles = vals_deg
                    # print("[Reader] angles update:", vals_deg)

            except serial.SerialException as e:
                print(f"[Serial Error] {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[Reader Exception] {e}")
                time.sleep(0.1)

    def get_angles(self):
        return self.current_angles

    def send_angles(self, d1, d2, d3):
        """
        下发目标角度到 Arduino
        输入单位: 度 (float)，会转换为 int16 (0.01°)
        """
        try:
            vals = [int(round(d1*100)), int(round(d2*100)), int(round(d3*100))]
            payload = struct.pack("<3h", *vals)

            LEN = len(payload)  # 6
            frame = bytearray([0xAA, 0x55, LEN]) + payload

            chk = 0
            for b in frame[2:]:
                chk ^= b
            frame.append(chk)

            self.ser.write(frame)
            self.ser.flush()
            print(f"[PC] Sent target angles: {d1:.2f}, {d2:.2f}, {d3:.2f}")
        except Exception as e:
            print(f"[PC] Send error: {e}")

    def close(self):
        self._stop_flag = True
        time.sleep(0.05)
        try:
            self.ser.close()
        except Exception:
            pass

def stress_test():
    comm = ArduinoComm(
        port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        baudrate=115200,
        timeout=0.001
    )

    try:
        N = 2000   # 总共发多少帧
        send_hz = 200000  # 目标下发频率 (Hz)
        interval = 1.0 / send_hz

        sent_count = 0
        recv_count = 0
        t0 = time.time()

        while sent_count < N:
            # 发
            angs = np.random.uniform(40, 45, size=3)
            comm.send_angles(*angs)
            sent_count += 1

            # 收
            vals = comm.get_angles()
            if all(30.0 <= v <= 70.0 for v in vals):
                recv_count += 1

            time.sleep(interval)

        t1 = time.time()
        duration = t1 - t0
        print("=== 测试结果 ===")
        print(f"目标频率: {send_hz} Hz, 总时长: {duration:.3f} s")
        print(f"下发帧数: {sent_count}, 接收帧数: {recv_count}")
        print(f"平均下发频率: {sent_count/duration:.1f} Hz")
        print(f"平均接收频率: {recv_count/duration:.1f} Hz")

    finally:
        comm.close()

if __name__ == "__main__":
    stress_test()
