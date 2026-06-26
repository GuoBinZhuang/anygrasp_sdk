import os
import sys
import numpy as np
from PIL import Image
import torch
import time
from scipy.ndimage import label, binary_dilation

# 添加 contact_graspnet 路径
sys.path.append('/home/gb/My_respositories/contact_graspnet_pytorch')
sys.path.append('/home/gb/My_respositories/contact_graspnet_pytorch/contact_graspnet_pytorch')

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO 

def compute_qd_map(depth: np.ndarray, window_size: int = 5) -> np.ndarray:
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
    qd = q_depth_stat
    qd[depth == 0] = 0.0
    return qd

def cgn_grasp_nms(grasps, scores, contact_pts, translation_threshold=0.03, rotation_threshold_deg=30.0):
    if len(grasps) == 0:
        return grasps, scores, contact_pts
    idx = np.argsort(-scores)
    grasps = grasps[idx]
    scores = scores[idx]
    contact_pts = contact_pts[idx]
    
    keep = []
    disabled = np.zeros(len(grasps), dtype=bool)
    
    for i in range(len(grasps)):
        if disabled[i]:
            continue
        keep.append(i)
        
        t_i = grasps[i, :3, 3]
        t_others = grasps[i+1:, :3, 3]
        dist_t = np.linalg.norm(t_others - t_i, axis=1)
        
        R_i = grasps[i, :3, :3]
        R_others = grasps[i+1:, :3, :3]
        traces = np.einsum('ab,mba->m', R_i, R_others)
        cos_ang = (traces - 1.0) / 2.0
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        dist_r = np.degrees(np.arccos(cos_ang))
        
        conflict = (dist_t < translation_threshold) & (dist_r < rotation_threshold_deg)
        disabled[i+1:][conflict] = True
        
    return grasps[keep], scores[keep], contact_pts[keep]

def main():
    print("=== 初始化 TransCG 数据集 (Scene 1) 抓取评测 ===")
    
    ckpt_dir = '/home/gb/My_respositories/contact_graspnet_pytorch/checkpoints/contact_graspnet'
    global_config = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=[])
    grasp_estimator = GraspEstimator(global_config)

    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    checkpoint_io.load('model.pt')
    print("✓ Contact-GraspNet 权重加载成功！")
    
    dataset_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection/transcg_dataset/transcg"
    scene_dir = os.path.join(dataset_dir, "scene1")
    
    # 载入两个相机的内参
    cam_K_d435 = np.load(os.path.join(dataset_dir, "camera_intrinsics/1-camIntrinsics-D435.npy"))
    cam_K_l515 = np.load(os.path.join(dataset_dir, "camera_intrinsics/2-camIntrinsics-L515.npy"))
    
    # 动态获取 scene1 目录下所有的有效视角子目录
    viewpoints = sorted([d for d in os.listdir(scene_dir) if d.isdigit()], key=int)
    print(f"检测到 {len(viewpoints)} 个有效视角子目录，开始全量批量评估...")
    
    results = {
        "native_precisions": [],
        "native_counts": [],
        "proposed_precisions": [],
        "proposed_counts": []
    }
    
    processed_count = 0
    t_start = time.time()
    
    for vp in viewpoints:
        vp_dir = os.path.join(scene_dir, vp)
        
        # 自适应选择相机：优先 D435 (1), 如果没有则使用 L515 (2)
        if os.path.exists(os.path.join(vp_dir, "rgb1.png")):
            cam_idx = "1"
            cam_K = cam_K_d435
        elif os.path.exists(os.path.join(vp_dir, "rgb2.png")):
            cam_idx = "2"
            cam_K = cam_K_l515
        else:
            continue
            
        rgb_path = os.path.join(vp_dir, f"rgb{cam_idx}.png")
        depth_path = os.path.join(vp_dir, f"depth{cam_idx}.png")
        gt_depth_path = os.path.join(vp_dir, f"depth{cam_idx}-gt.png")
        gt_mask_path = os.path.join(vp_dir, f"depth{cam_idx}-gt-mask.png")
        
        if not os.path.exists(depth_path) or not os.path.exists(gt_depth_path) or not os.path.exists(gt_mask_path):
            continue
            
        try:
            rgb = np.array(Image.open(rgb_path))
            depth_raw = np.array(Image.open(depth_path)) # 16-bit 毫米
            gt_depth = np.array(Image.open(gt_depth_path)) # 16-bit 毫米
            gt_mask = np.array(Image.open(gt_mask_path)) > 0
            H, W = depth_raw.shape
            
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            cx, cy = cam_K[0, 2], cam_K[1, 2]
            
            # 深度转为米传入模型 (CGN 要求米单位)
            depth_raw_m = depth_raw.astype(np.float32) / 1000.0
            
            # ==========================================
            # 1. 方案一：Native Baseline
            # ==========================================
            pc_full_nat, pc_segments_nat, _ = grasp_estimator.extract_point_clouds(
                depth_raw_m, cam_K, segmap=gt_mask.astype(np.int32), rgb=rgb, z_range=[0.2, 1.8]
            )
            
            # 预测
            grasps_nat, scores_nat, contact_pts_nat, _ = grasp_estimator.predict_scene_grasps(
                pc_full_nat, pc_segments=pc_segments_nat, local_regions=True, filter_grasps=True, forward_passes=1
            )
            
            obj_key = 1.0
            if obj_key in grasps_nat and len(grasps_nat[obj_key]) > 0:
                g_nat, s_nat, c_nat = grasps_nat[obj_key], scores_nat[obj_key], contact_pts_nat[obj_key]
                g_nat, s_nat, c_nat = cgn_grasp_nms(g_nat, s_nat, c_nat)
                g_nat, s_nat, c_nat = g_nat[:50], s_nat[:50], c_nat[:50]
            else:
                g_nat, c_nat = [], []
                
            # 评估 Native
            native_valid = 0
            for pt in c_nat:
                x, y, z = pt
                u = int(np.round((x * fx) / z + cx))
                v = int(np.round((y * fy) / z + cy))
                if 0 <= u < W and 0 <= v < H:
                    if gt_mask[v, u]:
                        z_gt = gt_depth[v, u] / 1000.0
                        if z_gt > 0 and abs(z - z_gt) < 0.03:
                            native_valid += 1
            
            native_precision = native_valid / len(c_nat) if len(c_nat) > 0 else 0.0
            
            # ==========================================
            # 2. 方案二：Proposed Pipeline
            # ==========================================
            qd = compute_qd_map(depth_raw)
            
            depth_mm = depth_raw.astype(np.float32)
            dy, dx = np.gradient(depth_mm)
            grad_mag = np.sqrt(dx**2 + dy**2)
            jump_mask = (grad_mag > 15.0) | (depth_raw == 0)
            jump_mask_expanded = binary_dilation(jump_mask, structure=np.ones((3, 3)))
            
            valid_mask_raw = gt_mask & (qd > 0.0)
            valid_mask_isolated = valid_mask_raw & (~jump_mask_expanded)
            
            valid_mask = np.zeros_like(gt_mask, dtype=bool)
            if np.any(valid_mask_isolated):
                labeled, num_features = label(valid_mask_isolated)
                if num_features > 0:
                    bincounts = np.bincount(labeled.ravel())
                    largest_label = np.argmax(bincounts[1:]) + 1
                    largest_cc = (labeled == largest_label)
                    valid_mask = binary_dilation(largest_cc, structure=np.ones((5, 5)))
                    
            depth_processed_m = depth_raw_m.copy()
            depth_processed_m[~valid_mask] = 0.0
            
            # 提取点云
            pc_full_prop, pc_segments_prop, _ = grasp_estimator.extract_point_clouds(
                depth_processed_m, cam_K, segmap=valid_mask.astype(np.int32), rgb=rgb, z_range=[0.2, 1.8]
            )
            
            # 预测
            grasps_prop, scores_prop, contact_pts_prop, _ = grasp_estimator.predict_scene_grasps(
                pc_full_prop, pc_segments=pc_segments_prop, local_regions=True, filter_grasps=True, forward_passes=1
            )
            
            if obj_key in grasps_prop and len(grasps_prop[obj_key]) > 0:
                g_prop, s_prop, c_prop = grasps_prop[obj_key], scores_prop[obj_key], contact_pts_prop[obj_key]
                g_prop, s_prop, c_prop = cgn_grasp_nms(g_prop, s_prop, c_prop)
                g_prop, s_prop, c_prop = g_prop[:50], s_prop[:50], c_prop[:50]
            else:
                g_prop, c_prop = [], []
                
            # 评估 Proposed
            proposed_valid = 0
            for pt in c_prop:
                x, y, z = pt
                u = int(np.round((x * fx) / z + cx))
                v = int(np.round((y * fy) / z + cy))
                if 0 <= u < W and 0 <= v < H:
                    if gt_mask[v, u]:
                        z_gt = gt_depth[v, u] / 1000.0
                        if z_gt > 0 and abs(z - z_gt) < 0.03:
                            proposed_valid += 1
                            
            proposed_precision = proposed_valid / len(c_prop) if len(c_prop) > 0 else 0.0
            
            results["native_precisions"].append(native_precision)
            results["native_counts"].append(len(c_nat))
            results["proposed_precisions"].append(proposed_precision)
            results["proposed_counts"].append(len(c_prop))
            
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == len(viewpoints):
                elapsed = time.time() - t_start
                print(f"  [进度] 已评估 {processed_count}/{len(viewpoints)} 个视角, 耗时 {elapsed:.1f}s")
                
        except Exception as e:
            # 捕获异常，防止批量测试中断
            print(f"  ⚠ 视角 {vp} 评估失败: {e}")
            continue

    # 汇总统计
    if len(results["native_precisions"]) > 0:
        mean_nat_prec = np.mean(results["native_precisions"])
        std_nat_prec = np.std(results["native_precisions"])
        mean_nat_cnt = np.mean(results["native_counts"])
        
        mean_prop_prec = np.mean(results["proposed_precisions"])
        std_prop_prec = np.std(results["proposed_precisions"])
        mean_prop_cnt = np.mean(results["proposed_counts"])
        
        print("\n================ 评测汇总结果 ================")
        print(f"测试样本视角总量 (N) | {processed_count}")
        print("Method | Target Mask 3D Alignment Precision | Average Candidate Count")
        print("---------------------------------------------------------------------")
        print(f"Baseline (CGN-Mask on Raw) | {mean_nat_prec:.4f} ± {std_nat_prec:.4f} | {mean_nat_cnt:.1f}")
        print(f"Proposed (Our filter pipeline) | {mean_prop_prec:.4f} ± {std_prop_prec:.4f} | {mean_prop_cnt:.1f}")
        print("==============================================")
    else:
        print("❌ 未成功评估任何有效视角")

if __name__ == "__main__":
    main()
