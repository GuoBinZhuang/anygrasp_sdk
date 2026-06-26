import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from rosbags.highlevel import AnyReader

COLOR_TOPIC = "/camera/camera/color/image_raw"
ALIGNED_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/aligned_depth_to_color/camera_info"

def image_to_numpy(msg):
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "bgr8":
            arr = arr[..., ::-1]
        return arr.copy()
    elif msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).copy()
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def extract_from_single_bag(bag_path, output_dir, start_idx, sample_count=3, max_sync_delta_ms=80.0):
    print(f"\n--- Processing bag: {bag_path.name} ---")
    color_frames = []
    depth_frames = []
    camera_info = None

    try:
        with AnyReader([bag_path]) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == COLOR_TOPIC:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    arr = image_to_numpy(msg)
                    color_frames.append({
                        "timestamp": timestamp,
                        "array": arr
                    })
                elif connection.topic == ALIGNED_DEPTH_TOPIC:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    arr = image_to_numpy(msg)
                    depth_frames.append({
                        "timestamp": timestamp,
                        "array": arr
                    })
                elif connection.topic == CAMERA_INFO_TOPIC:
                    if camera_info is None:
                        camera_info = reader.deserialize(rawdata, connection.msgtype)
    except Exception as e:
        print(f"❌ Error opening or reading bag {bag_path.name}: {e}")
        return 0

    print(f"  Color frames: {len(color_frames)} | Depth frames: {len(depth_frames)}")
    if not color_frames or not depth_frames:
        print(f"  ⚠ Skip: Missing color or depth frames in {bag_path.name}")
        return 0

    # 解析相机内参
    intrinsics = None
    if camera_info is not None:
        k = camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        intrinsics = (fx, fy, cx, cy)
        print(f"  ✓ Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 最邻近同步
    synced_pairs = []
    for c_frame in color_frames:
        nearest_d = min(depth_frames, key=lambda d: abs(d["timestamp"] - c_frame["timestamp"]))
        delta_ms = abs(nearest_d["timestamp"] - c_frame["timestamp"]) / 1e6
        if delta_ms <= max_sync_delta_ms:
            synced_pairs.append((c_frame, nearest_d))

    print(f"  Synchronized pairs: {len(synced_pairs)}")
    if not synced_pairs:
        print(f"  ⚠ Skip: No synchronized frames in {bag_path.name}")
        return 0

    # 均匀采样
    positions = np.linspace(0.15, 0.85, sample_count)
    selected_indices = [int(round(pos * (len(synced_pairs) - 1))) for pos in positions]
    selected_indices = sorted(list(set(selected_indices)))

    extracted_count = 0
    for offset_idx, select_idx in enumerate(selected_indices):
        c_frame, d_frame = synced_pairs[select_idx]
        global_idx = start_idx + extracted_count
        sample_dir = output_dir / f"sample_{global_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        color_img = Image.fromarray(c_frame["array"])
        depth_img = Image.fromarray(d_frame["array"])

        color_img.save(sample_dir / "color.png")
        depth_img.save(sample_dir / "depth.png")

        # 写入该 sample 的内参
        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
            with open(sample_dir / "intrinsics.txt", "w") as f:
                f.write(f"{fx} {fy} {cx} {cy} 1000.0\n")

        print(f"    Saved sample_{global_idx:02d} (from synced pair {select_idx})")
        extracted_count += 1

    return extracted_count

def main():
    parser = argparse.ArgumentParser(description="Batch extract synchronized RGB-D frames from multiple MCAP bags.")
    parser.add_argument("--bag-dir", type=str, required=True, help="Parent directory containing multiple mcap folders")
    parser.add_argument("--output-dir", type=str, default="./tfb_extracted_data", help="Output directory")
    parser.add_argument("--sample-per-bag", type=int, default=3, help="Number of frames to extract per bag")
    args = parser.parse_args()

    parent_bag_dir = Path(args.bag_dir)
    output_dir = Path(args.output_dir)
    
    # 清理并重建输出目录，保证独立样本纯净
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描子文件夹
    mcap_dirs = sorted([
        d for d in parent_bag_dir.iterdir()
        if d.is_dir() and (d.name.endswith(".mcap") or (d / "metadata.yaml").exists())
    ])

    print(f"Found {len(mcap_dirs)} candidate bags in {parent_bag_dir}")
    
    global_extracted_count = 0
    for mcap_dir in mcap_dirs:
        count = extract_from_single_bag(
            mcap_dir, 
            output_dir, 
            global_extracted_count, 
            sample_count=args.sample_per_bag
        )
        global_extracted_count += count

    print(f"\n🎉 Extraction finished! Total extracted independent samples: {global_extracted_count}")

if __name__ == '__main__':
    main()
