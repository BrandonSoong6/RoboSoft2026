import serial
import threading
import time
import struct

class ArduinoComm:
    def __init__(self,
                 port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                 baudrate=115200,
                 timeout=0.05):  # 适中超时
        # 打开串口
        self.ser = serial.Serial(
            port, baudrate=baudrate, timeout=timeout, write_timeout=0.5
        )
        time.sleep(1.0)  # 给 Arduino 复位时间
        self._stop_flag = False
        self.current_angles = (0.0, 0.0, 0.0)
        self._lock = threading.Lock()

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
                        # 没有帧头，清掉垃圾
                        buf = b""
                        break
                    if len(buf) < start + 3:
                        # 数据不够：至少要 header+len
                        break
                    length = buf[start+2]
                    frame_len = 2 + 1 + length + 1  # header + len + data + chk
                    if len(buf) < start + frame_len:
                        # 等待更多字节
                        break

                    frame = buf[start:start+frame_len]
                    buf = buf[start+frame_len:]  # 剩余数据

                    # 校验
                    chk_calc = 0
                    for b in frame[2:-1]:
                        chk_calc ^= b
                    chk_recv = frame[-1]
                    if chk_calc != chk_recv:
                        # 校验失败，丢弃
                        continue

                    data_bytes = frame[3:-1]  # 提取有效载荷
                    if len(data_bytes) % 2 != 0:
                        continue

                    # 解码小端 int16_t → 转成角度
                    vals_raw = struct.unpack("<" + "h"*(len(data_bytes)//2), data_bytes)
                    vals_deg = tuple(v/100.0 for v in vals_raw)

                    with self._lock:
                        self.current_angles = vals_deg

            except serial.SerialException as e:
                print(f"[Serial Error] {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[Reader Exception] {e}")
                time.sleep(0.1)

    def get_angles(self):
        """线程安全获取角度 (°)"""
        with self._lock:
            return self.current_angles

    def send_cmd(self, cmd: str):
        """发送命令到 Arduino（字符串命令接口）"""
        try:
            line = cmd.strip() + "\r\n"
            self.ser.write(line.encode("ascii"))
            print(f"[PC] Sent: {cmd}")
        except serial.SerialTimeoutException:
            print("[PC] Send timeout")
        except Exception as e:
            print(f"[PC] Send error: {e}")

    def close(self):
        self._stop_flag = True
        time.sleep(0.05)
        try:
            self.ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    comm = ArduinoComm()
    try:
        print("输入命令发送给Arduino，例如：drag / sync 120 130 140 / move 1 150")
        while True:
            print(f"[Arduino] 当前角度: {comm.get_angles()}")
            # cmd = input(">>> ")
            # if not cmd.strip():
            #     continue
            # comm.send_cmd(cmd)
    except KeyboardInterrupt:
        print("退出程序")
    finally:
        comm.close()
