import os
import sys
import numpy as np
from PIL import Image
import torch
import time
from scipy.ndimage import label, binary_dilation

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
        roi_mask[H//12 : 11*H//12, W//12 : 11*W//12] = True
        
        # 物理区间过滤：Z 限制在 [470mm, 780mm] 衣服实体区间
        # 既排除近处机械爪 (Z<=460mm)，又排除远处桌底背景
        physical_mask = (depth_np >= 470) & (depth_np <= 780)
        
        # 1. 蓝色多模态提取 (B > R * 1.15) & (G > R * 1.05) & (B > 50)
        R = color_np[:, :, 0].astype(np.float32)
        G = color_np[:, :, 1].astype(np.float32)
        B = color_np[:, :, 2].astype(np.float32)
        blue_mask = (B > R * 1.15) & (G > R * 1.05) & (B > 50)
        
        # 2. 深度跳变
        depth_mm = depth_np.astype(np.float32)
        dy, dx = np.gradient(depth_mm)
        grad_mag = np.sqrt(dx**2 + dy**2)
        jump_mask = (grad_mag > 20.0) | (depth_np == 0)
        jump_mask_expanded = binary_dilation(jump_mask, structure=np.ones((3, 3)))
        
        # 3. 计算 valid_mask_raw 并扣除蓝色箱体和跳变边缘
        valid_mask_raw = roi_mask & physical_mask & (qd > 0.0)
        valid_mask_isolated = valid_mask_raw & (~jump_mask_expanded) & (~blue_mask)
        
        # 4. 最大连通域提取
        valid_mask = np.zeros_like(valid_mask_raw, dtype=bool)
        if np.any(valid_mask_isolated):
            labeled_mask, num_features = label(valid_mask_isolated)
            if num_features > 0:
                bincounts = np.bincount(labeled_mask.ravel())
                largest_label = np.argmax(bincounts[1:]) + 1
                largest_cc = (labeled_mask == largest_label)
                valid_mask = binary_dilation(largest_cc, structure=np.ones((3, 3)))
        
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
            
            # 计算 Box 并进行安全外扩 (35像素)
            x_min_raw = int(xs.min())
            x_max_raw = int(xs.max())
            y_min_raw = int(ys.min())
            y_max_raw = int(ys.max())
            
            x_min = 0 if x_min_raw < 30 else max(0, x_min_raw - 35)
            x_max = W - 1 if x_max_raw > W - 30 else min(W - 1, x_max_raw + 35)
            y_min = 0 if y_min_raw < 30 else max(0, y_min_raw - 35)
            y_max = H - 1 if y_max_raw > H - 30 else min(H - 1, y_max_raw + 35)
            
            input_box = np.array([x_min, y_min, x_max, y_max])
        else:
            u, v = W // 2, H // 2
            # 回退 Box 为中心的大方框
            input_box = np.array([W//4, H//4, 3*W//4, 3*H//4])
            
        # 生成负样本点提示，背景包含：深度背景 (>= 790mm) 以及 蓝色箱体 (blue_mask)
        bg_mask = (depth_np >= 790) | blue_mask
        pts_list = [[u, v]]
        labels_list = [1] # 1 = 前景
        
        # 自适应候选背景点 (上下左右距离物体 55 像素的区域，防止干涉外拓后的 Box)
        if input_box is not None:
            bx_min, by_min, bx_max, by_max = input_box
            candidates_bg = [
                (max(0, bx_min - 55), (by_min + by_max)//2),
                (min(W - 1, bx_max + 55), (by_min + by_max)//2),
                ((bx_min + bx_max)//2, max(0, by_min - 55)),
                ((bx_min + bx_max)//2, min(H - 1, by_max + 55))
            ]
            for cx, cy in candidates_bg:
                # 只有当该点确属背景或深度无效区时才作为背景负点
                if bg_mask[cy, cx] or depth_np[cy, cx] == 0:
                    pts_list.append([cx, cy])
                    labels_list.append(0) # 0 = 背景
                    
        input_points = np.array(pts_list)
        input_labels = np.array(labels_list)
        
        # 进行第一阶段 SAM2 分割推理
        predictor.set_image(color_np)
        
        t0 = time.time()
        # multimask_output=True 可以获取多层次的 mask，我们选择得分最高的一个
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        mask_stage1 = masks[best_idx]
        if torch.is_tensor(mask_stage1):
            mask_stage1 = mask_stage1.cpu().numpy()
        mask_stage1 = mask_stage1.astype(bool)
        
        # 二轮迭代自适应修正 (Two-stage Iterative Refinement)
        refined_points = list(pts_list)
        refined_labels = list(labels_list)
        
        # 1. 寻找漏分的前景区 (False Negatives): 属于 valid_mask (包含 LCC，不包含蓝色和跳变)
        # 排除 blue_mask 确保漏分点绝对不会点在蓝色箱体上
        fg_physical = roi_mask & physical_mask & (qd > 0.05) & (~blue_mask)
        fn_mask = fg_physical & ~mask_stage1
        
        # 2. 寻找溢出的背景区 (False Positives): 属于 bg_mask 却在第一阶段被错误割入
        fp_mask = bg_mask & mask_stage1
        
        updated_prompts = False
        # 如果漏分像素过多（超过 150 像素），则提取其高 Qd 像素进行正点补加
        if np.any(fn_mask) and np.sum(fn_mask) > 150:
            ys_fn, xs_fn = np.where(fn_mask)
            qd_fn = qd[ys_fn, xs_fn]
            best_fn_idx = np.argmax(qd_fn)
            fn_u, fn_v = int(xs_fn[best_fn_idx]), int(ys_fn[best_fn_idx])
            refined_points.append([fn_u, fn_v])
            refined_labels.append(1) # 1 = 正样本
            updated_prompts = True
            
        # 如果背景溢出像素过多（超过 150 像素），则在误分处补加负点
        if np.any(fp_mask) and np.sum(fp_mask) > 150:
            ys_fp, xs_fp = np.where(fp_mask)
            fp_u = int(np.round(np.mean(xs_fp)))
            fp_v = int(np.round(np.mean(ys_fp)))
            
            dists = (xs_fp - fp_u)**2 + (ys_fp - fp_v)**2
            nearest_fp = np.argmin(dists)
            fp_u_real = int(xs_fp[nearest_fp])
            fp_v_real = int(ys_fp[nearest_fp])
            
            refined_points.append([fp_u_real, fp_v_real])
            refined_labels.append(0) # 0 = 背景
            updated_prompts = True
            
        # 如果需要二轮修正，重新进行推理
        if updated_prompts:
            input_points_final = np.array(refined_points)
            input_labels_final = np.array(refined_labels)
            
            masks_refined, scores_refined, _ = predictor.predict(
                point_coords=input_points_final,
                point_labels=input_labels_final,
                box=input_box,
                multimask_output=True
            )
            best_idx = np.argmax(scores_refined)
            best_mask = masks_refined[best_idx]
            if torch.is_tensor(best_mask):
                best_mask = best_mask.cpu().numpy()
            best_mask = best_mask.astype(bool)
            final_score = scores_refined[best_idx]
        else:
            best_mask = mask_stage1
            input_points_final = input_points
            input_labels_final = input_labels
            final_score = scores[best_idx]
            
        elapsed = time.time() - t0
        
        # 保存真正的 sam2_mask.npy
        mask_save_path = os.path.join(sample_path, "sam2_mask.npy")
        np.save(mask_save_path, best_mask)
        
        # 保存可视化图像对比
        overlay = color_np.copy()
        overlay[best_mask] = (overlay[best_mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        
        from PIL import ImageDraw
        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)
        
        # 1. 绘制初始前景点 (红十字)
        draw.line((u - 5, v, u + 5, v), fill="red", width=2)
        draw.line((u, v - 5, u, v + 5), fill="red", width=2)
        
        # 2. 绘制自适应背景负点和二次修正的红/蓝十字
        for pt, lbl in zip(input_points_final[1:], input_labels_final[1:]):
            bx, by = pt
            fill_color = "red" if lbl == 1 else "blue"
            draw.line((bx - 4, by, bx + 4, by), fill=fill_color, width=2)
            draw.line((bx, by - 4, bx, by + 4), fill=fill_color, width=2)
            
        # 3. 绘制 Bounding Box (红色线框)
        if input_box is not None:
            bx_min, by_min, bx_max, by_max = input_box
            draw.rectangle([bx_min, by_min, bx_max, by_max], outline="red", width=2)
            
        vis_save_path = os.path.join(sample_path, "sam2_mask_vis.png")
        overlay_pil.save(vis_save_path)
        
        print(f"  ✓ 样本 {sample} 掩码已生成，耗时 {elapsed:.2f}s, 提示点数: {len(input_points_final)} (红正蓝负, 修正: {updated_prompts}), score: {final_score:.4f}")
        
    print("\n🎉 全部样本的真实 SAM2.1 掩码处理完成！")

if __name__ == '__main__':
    main()
