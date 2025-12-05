import asyncio
import websockets
import json
import numpy as np
from vispy import scene

# ===============================
#  VISPY — 3D Skeleton Viewer
# ===============================
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Define joint connections (you can extend this based on Rokoko joint names)
JOINT_CONNECTIONS = [
    ("hips", "spine"),
    ("spine", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("hips", "left_knee"),
    ("left_knee", "left_foot"),
    ("hips", "right_knee"),
    ("right_knee", "right_foot"),
]

# Create a placeholder line series
lines = scene.Line(
    pos=np.zeros((len(JOINT_CONNECTIONS)*2, 3)),
    color='cyan',
    width=3,
    parent=view.scene
)

# File to store all JSON V3 events
SAVE_FILE = "rokoko_jsonv3_stream.jsonl"
fout = open(SAVE_FILE, "w", encoding="utf-8")


# ===============================
#  STREAM FUNCTION (JSON V3)
# ===============================
async def stream_v3():
    uri = "ws://127.0.0.1:14043"

    print(f"🔗 Connecting to Rokoko JSON V3 WebSocket at {uri} ...")
    async with websockets.connect(uri) as ws:
        print("✅ Connected to Rokoko JSON V3 WebSocket!")
        print("⏳ Waiting for motion data...\n")

        joint_positions = {}

        while True:
            raw = await ws.recv()
            fout.write(raw + "\n")  # save JSON V3 line

            data = json.loads(raw)

            # Only update if this is a frame containing joints
            if data.get("type") == "body.frame":
                joints = data.get("joints", [])

                # Build mapping joint → XYZ
                for j in joints:
                    name = j["name"]
                    pos  = j.get("position", [0, 0, 0])
                    joint_positions[name] = pos

                # Build list of connected line segments
                positions = []
                for a, b in JOINT_CONNECTIONS:
                    if a in joint_positions and b in joint_positions:
                        positions.append(joint_positions[a])
                        positions.append(joint_positions[b])

                if positions:
                    lines.set_data(np.array(positions))

                canvas.update()


# ===============================
#  RUN STREAMING
# ===============================
if __name__ == "__main__":
    try:
        asyncio.run(stream_v3())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        fout.close()
        print(f"💾 JSON V3 data saved to {SAVE_FILE}")
