from old_mocap_module import UDPClient, MotionSuitReceiver
import json
import time

# Create UDPClient as buffer
client = UDPClient(dev_mode=False)

import asyncio
import websockets
import json
import threading

class WebSocketReceiver:
    def __init__(self, ip="127.0.0.1", port=14053, api_key="1234", udp_client=None):
        self.url = f"ws://{ip}:{port}"
        self.api_key = api_key
        self.udp_client = udp_client  # forward vào UDPClient queue
        self.running = True

    async def listen(self):
        async with websockets.connect(
            self.url,
            extra_headers={"api-key": self.api_key},
            max_size=2**24  # allow large JSON frames
        ) as ws:

            print("Connected to Rokoko WebSocket JSON v3")

            while self.running:
                try:
                    msg = await ws.recv()
                    if self.udp_client:
                        with self.udp_client.lock:
                            self.udp_client.queue.put(msg)
                except Exception as e:
                    print("WS error:", e)
                    await asyncio.sleep(1)

    def start(self):
        thread = threading.Thread(target=lambda: asyncio.run(self.listen()), daemon=True)
        thread.start()


ws = WebSocketReceiver(
    ip="127.0.0.1",
    port=14053,
    api_key="1234",
    udp_client=client
)

ws.start()

print("Listening for Rokoko JSON v3 frames...")

while True:
    data = client.update_cache()
    if data:
        frame = json.loads(data)
        print("Received frame type:", frame.get("type"))

    time.sleep(1/25)  # 25 FPS

