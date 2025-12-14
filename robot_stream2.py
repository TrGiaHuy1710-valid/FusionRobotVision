import socket
import json
import time

# Cấu hình phải khớp với Rokoko Studio
UDP_IP = "192.168.56.1"
UDP_PORT = 14043
OUTPUT_FILE = "rokoko_motion_log_v3.json"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
sock.bind((UDP_IP, UDP_PORT))

print(f"Đang lắng nghe dữ liệu từ Rokoko tại {UDP_IP}:{UDP_PORT}...")

motion_logs = []

try:
    while True:
        # Nhận dữ liệu (buffer size 65535 là đủ lớn cho 1 frame JSON)
        data, addr = sock.recvfrom(65535)

        # Decode dữ liệu từ bytes sang string rồi sang JSON object
        json_str = data.decode('utf-8')
        try:
            json_data = json.loads(json_str)

            # Thêm timestamp nếu cần để đồng bộ sau này
            json_data['capture_timestamp'] = time.time()

            # In thử ra màn hình để check
            # print(json.dumps(json_data, indent=2)) 

            motion_logs.append(json_data)
            print(f"Đã nhận frame: {len(motion_logs)}")

        except json.JSONDecodeError:
            print("Lỗi decode JSON gói tin.")

except KeyboardInterrupt:
    # Nhấn Ctrl+C để dừng và lưu file
    print("\nĐang dừng và lưu file...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(motion_logs, f, indent=4)
    print(f"Đã lưu {len(motion_logs)} frames vào file {OUTPUT_FILE}")

finally:
    sock.close()