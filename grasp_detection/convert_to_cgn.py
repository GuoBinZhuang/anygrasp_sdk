import os
import numpy as np
from PIL import Image

def load_intrinsics(sample_path):
    intrinsics_path = os.path.join(sample_path, 'intrinsics.txt')
    with open(intrinsics_path, 'r') as f:
        line = f.readline().strip()
        parts = [float(x) for x in line.split()]
        # fx, fy, cx, cy, scale
        return parts[0], parts[1], parts[2], parts[3], parts[4]

def main():
    data_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection/tfb_extracted_data"
    out_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection/cgn_input_data"
    os.makedirs(out_dir, exist_ok=True)
    
    samples = sorted([d for d in os.listdir(data_dir) if d.startswith("sample_")])
    print(f"开始转换，检测到 {len(samples)} 个样本")
    
    for sample in samples:
        sample_path = os.path.join(data_dir, sample)
        
        # 1. 读取 color, depth, sam2_mask
        color_path = os.path.join(sample_path, 'color.png')
        depth_path = os.path.join(sample_path, 'depth.png')
        sam2_mask_path = os.path.join(sample_path, 'sam2_mask.npy')
        
        if not (os.path.exists(color_path) and os.path.exists(depth_path) and os.path.exists(sam2_mask_path)):
            print(f"⚠ 样本 {sample} 文件缺失，跳过！")
            continue
            
        rgb = np.array(Image.open(color_path)) # H x W x 3, uint8
        depth_raw = np.array(Image.open(depth_path)) # H x W, uint16
        sam2_mask = np.load(sam2_mask_path) # H x W bool
        
        # 2. 读取内参并转换深度
        fx, fy, cx, cy, scale = load_intrinsics(sample_path)
        depth_m = depth_raw.astype(np.float32) / scale
        
        # 构造 K
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 3. 将 segmap 转为 int32 (1 代表目标物体, 0 代表背景)
        segmap = sam2_mask.astype(np.int32)
        
        # 4. 保存为 npz 文件
        out_path = os.path.join(out_dir, f"{sample}.npz")
        np.savez(out_path, depth=depth_m, K=K, segmap=segmap, rgb=rgb)
        print(f"✓ 已生成 {out_path}: depth shape={depth_m.shape}, K_flat={K.flatten().tolist()}, sam2_ratio={np.mean(sam2_mask):.4f}")

if __name__ == "__main__":
    main()
