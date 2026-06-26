import os
import sys
import numpy as np
from PIL import Image
import torch
import time

# 将 sam2 目录加入 path，必须用 insert(0, ...) 确保优先级最高
sys.path.insert(0, "/home/gb/My_respositories/OAS_BagSeg/sam2")
from sam2.build_sam import build_sam2
import sam2.build_sam
print("Loaded build_sam from:", sam2.build_sam.__file__)
from sam2.sam2_image_predictor import SAM2ImagePredictor

def compute_qd_map(depth: np.ndarray, ir_intensity: np.ndarray = None, window_size: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    depth_f = depth.astype(np.float32)
    valid = (depth > 0).astype(np.float32)
    
    local_sum = uniform_filter(depth_f * valid, size=window_size)
    local_cnt = uniform_filter(valid, size=window_size) + 1e-6
    local_mean = local_sum / local_cnt

    local_sq_sum = uniform_filter((depth_f ** 2) * valid, size=window_size)
    local_sq_mean = local_sq_sum / local_cnt
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    std_max = np.percentile(local_std[valid > 0], 95) + 1e-6
    q_depth_stat = 1.0 - np.clip(local_std / std_max, 0, 1)

    if ir_intensity is not None:
        ir_f = ir_intensity.astype(np.float32)
        ir_max = np.percentile(ir_f, 99) + 1e-6
        q_ir = np.clip(ir_f / ir_max, 0, 1)
        qd = 0.6 * q_depth_stat + 0.4 * q_ir
    else:
        qd = q_depth_stat

    qd[depth == 0] = 0.0
    return qd.astype(np.float32)

def main():
    import os
    os.chdir("/home/gb/My_respositories/anygrasp_sdk/grasp_detection")
    print("=== 初始化真实 SAM2.1 Large 推理引擎 ===")
    model_cfg = "sam2_hiera_l.yaml" # 2.1 大模型配置名称 
    checkpoint = "/home/gb/My_respositories/OAS_BagSeg/sam2/checkpoints/sam2.1_hiera_large.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {device}")
    
    try:
        model = build_sam2(model_cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(model)
        print("  ✓ SAM2.1 Large 模型与权重加载成功！")
    except Exception as e:
        print(f"❌ 加载 SAM2 失败: {e}")
        return

    data_dir = "./tfb_extracted_data"
    samples = sorted([d for d in os.listdir(data_dir) if d.startswith("sample_")])
    
    print(f"\n开始为 {len(samples)} 个独立样本生成真实 SAM2 掩码...")
    
    for sample in samples:
        sample_path = os.path.join(data_dir, sample)
        color_path = os.path.join(sample_path, "color.png")
        depth_path = os.path.join(sample_path, "depth.png")
        
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"  ⚠ 样本 {sample} 缺少图像，跳过")
            continue
            
        color_img = Image.open(color_path)
        color_np = np.array(color_img)
        depth_np = np.array(Image.open(depth_path))
        
        H, W = depth_np.shape[:2]
        
        # 计算 Qd 图
        qd = compute_qd_map(depth_np)
        
        # 限制在中心区域 ROI 寻找最可靠点 (x: [W/4, 3W/4], y: [H/4, 3H/4])
        roi_mask = np.zeros_like(qd, dtype=bool)
        roi_mask[H//4 : 3*H//4, W//4 : 3*W//4] = True
        
        # 物理区间过滤：Z 限制在 [490mm, 760mm] 衣服实体区间
        # 既排除近处机械爪 (Z<=480mm)，又排除桌面背景 (Z>=770mm)
        physical_mask = (depth_np >= 490) & (depth_np <= 760)
        
        # 提取 top 10% 最高 Qd 置信度像素的加权质心，阻断极值噪声和几何拉拽伪影
        valid_mask = roi_mask & physical_mask & (qd > 0.0)
        input_box = None
        if np.any(valid_mask):
            ys, xs = np.where(valid_mask)
            qd_vals = qd[ys, xs]
            sort_idx = np.argsort(qd_vals)[::-1]
            num_to_keep = max(100, int(np.ceil(len(sort_idx) * 0.10)))
            num_to_keep = min(num_to_keep, len(sort_idx))
            
            keep_idx = sort_idx[:num_to_keep]
            ys_keep = ys[keep_idx]
            xs_keep = xs[keep_idx]
            weights = qd_vals[keep_idx]
            
            u_raw = int(np.round(np.sum(xs_keep * weights) / np.sum(weights)))
            v_raw = int(np.round(np.sum(ys_keep * weights) / np.sum(weights)))
            
            dists = (xs - u_raw)**2 + (ys - v_raw)**2
            nearest_idx = np.argmin(dists)
            u = int(xs[nearest_idx])
            v = int(ys[nearest_idx])
            
            # 计算 Box 并进行安全外扩 (15像素)
            x_min = max(0, int(xs.min()) - 15)
            x_max = min(W - 1, int(xs.max()) + 15)
            y_min = max(0, int(ys.min()) - 15)
            y_max = min(H - 1, int(ys.max()) + 15)
            input_box = np.array([x_min, y_min, x_max, y_max])
        else:
            u, v = W // 2, H // 2
            # 回退 Box 为中心的大方框
            input_box = np.array([W//4, H//4, 3*W//4, 3*H//4])
            
        # 生成负样本点提示，抑制溢出到背景桌面 (Z >= 780mm)
        bg_mask = (depth_np >= 780)
        pts_list = [[u, v]]
        labels_list = [1] # 1 = 前景
        
        # 自适应候选背景点 (上下左右距离物体 45 像素的区域)
        if input_box is not None:
            bx_min, by_min, bx_max, by_max = input_box
            candidates_bg = [
                (max(0, bx_min - 45), (by_min + by_max)//2),
                (min(W - 1, bx_max + 45), (by_min + by_max)//2),
                ((bx_min + bx_max)//2, max(0, by_min - 45)),
                ((bx_min + bx_max)//2, min(H - 1, by_max + 45))
            ]
            for cx, cy in candidates_bg:
                # 只有当该点确属背景（Z >= 780）或深度无效区时才作为背景负点
                if bg_mask[cy, cx] or depth_np[cy, cx] == 0:
                    pts_list.append([cx, cy])
                    labels_list.append(0) # 0 = 背景
                    
        input_points = np.array(pts_list)
        input_labels = np.array(labels_list)
        
        # 进行 SAM2 分割推理
        predictor.set_image(color_np)
        
        t0 = time.time()
        # multimask_output=True 可以获取多层次的 mask，我们选择得分最高的一个
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True
        )
        elapsed = time.time() - t0
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        if torch.is_tensor(best_mask):
            best_mask = best_mask.cpu().numpy()
        best_mask = best_mask.astype(bool) # HxW bool
        
        # 保存真正的 sam2_mask.npy
        mask_save_path = os.path.join(sample_path, "sam2_mask.npy")
        np.save(mask_save_path, best_mask)
        
        # 保存可视化图像对比
        overlay = color_np.copy()
        overlay[best_mask] = (overlay[best_mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        
        from PIL import ImageDraw
        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)
        
        # 1. 绘制前景点 (红十字)
        draw.line((u - 5, v, u + 5, v), fill="red", width=2)
        draw.line((u, v - 5, u, v + 5), fill="red", width=2)
        
        # 2. 绘制自适应背景点 (蓝十字)
        for pt, lbl in zip(input_points[1:], input_labels[1:]):
            bx, by = pt
            draw.line((bx - 4, by, bx + 4, by), fill="blue", width=2)
            draw.line((bx, by - 4, bx, by + 4), fill="blue", width=2)
            
        # 3. 绘制 Bounding Box (红色线框)
        if input_box is not None:
            bx_min, by_min, bx_max, by_max = input_box
            draw.rectangle([bx_min, by_min, bx_max, by_max], outline="red", width=2)
            
        vis_save_path = os.path.join(sample_path, "sam2_mask_vis.png")
        overlay_pil.save(vis_save_path)
        
        print(f"  ✓ 样本 {sample} 掩码已生成，耗时 {elapsed:.2f}s, 提示点数: {len(input_points)} (红正蓝负), score: {scores[best_idx]:.4f}")
        
    print("\n🎉 全部样本的真实 SAM2.1 掩码处理完成！")

if __name__ == '__main__':
    main()
