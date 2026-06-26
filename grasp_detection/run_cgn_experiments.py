import os
import sys
import json
import numpy as np

# 添加 contact_graspnet 路径
sys.path.append('/home/gb/My_respositories/contact_graspnet_pytorch')
sys.path.append('/home/gb/My_respositories/contact_graspnet_pytorch/contact_graspnet_pytorch')

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from data import load_available_input_data

def cgn_grasp_nms(grasps, scores, contact_pts, translation_threshold=0.03, rotation_threshold_deg=30.0):
    if len(grasps) == 0:
        return grasps, scores, contact_pts
    # 按照 scores 从高到低排序
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
    print("=== 初始化 Contact-GraspNet 模型 ===")
    ckpt_dir = '/home/gb/My_respositories/contact_graspnet_pytorch/checkpoints/contact_graspnet'
    global_config = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=[])
    grasp_estimator = GraspEstimator(global_config)

    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    checkpoint_io.load('model.pt')
    print("✓ 模型和权重加载成功！")

    npz_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection/cgn_input_data"
    output_parent = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection/tfb_extracted_data"
    
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    print(f"找到 {len(npz_files)} 个数据样本，准备开始推理...")

    for f_name in npz_files:
        sample_name = f_name.replace(".npz", "")
        sample_npz_path = os.path.join(npz_dir, f_name)
        sample_output_dir = os.path.join(output_parent, sample_name)
        
        print(f"\n--- 处理样本 {sample_name} ---")
        
        # 1. 读取数据
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(sample_npz_path, K=None)
        
        # ==========================================
        # 实验1：Group CGN-Native (原生整场景无 mask)
        # ==========================================
        print(f" [{sample_name}] 运行 CGN-Native 推理...")
        # 提取整场景点云 (不传入 segmap)
        pc_full_native, _, _ = grasp_estimator.extract_point_clouds(
            depth, cam_K, segmap=None, rgb=rgb, z_range=[0.2, 1.8]
        )
        
        # 前向推理
        grasps_native, scores_native, contact_pts_native, _ = grasp_estimator.predict_scene_grasps(
            pc_full_native, pc_segments={}, local_regions=False, filter_grasps=False, forward_passes=1
        )
        
        # 提取结果并排序、NMS、截取 Top-50
        if -1 in grasps_native and len(grasps_native[-1]) > 0:
            g_nat, s_nat, c_nat = grasps_native[-1], scores_native[-1], contact_pts_native[-1]
            g_nat_nms, s_nat_nms, c_nat_nms = cgn_grasp_nms(g_nat, s_nat, c_nat)
            # 取 top-50
            g_nat_top = g_nat_nms[:50]
            s_nat_top = s_nat_nms[:50]
            c_nat_top = c_nat_nms[:50]
        else:
            g_nat_top, s_nat_top, c_nat_top = np.array([]), np.array([]), np.array([])
            
        # 保存实验结果
        res_native = {
            "group": "CGN_Native",
            "num_grasps": len(g_nat_top),
            "top1_score": float(s_nat_top[0]) if len(s_nat_top) > 0 else 0.0,
            "translations": g_nat_top[:, :3, 3].tolist() if len(g_nat_top) > 0 else [],
            "contact_pts": c_nat_top.tolist() if len(c_nat_top) > 0 else []
        }
        native_json_path = os.path.join(sample_output_dir, "result_group_CGN_Native.json")
        with open(native_json_path, 'w') as out_f:
            json.dump(res_native, out_f, indent=2)
        print(f"   ✓ CGN-Native 完成，获取 {len(g_nat_top)} 个抓取。已保存至 {native_json_path}")

        # ==========================================
        # 实验2：Group CGN-Mask (+mask 局部区域过滤)
        # ==========================================
        print(f" [{sample_name}] 运行 CGN-Mask 推理...")
        # 提取带 mask 分割的点云
        pc_full_mask, pc_segments_mask, _ = grasp_estimator.extract_point_clouds(
            depth, cam_K, segmap=segmap, rgb=rgb, z_range=[0.2, 1.8]
        )
        
        # 前向推理
        grasps_mask, scores_mask, contact_pts_mask, _ = grasp_estimator.predict_scene_grasps(
            pc_full_mask, pc_segments=pc_segments_mask, local_regions=True, filter_grasps=True, forward_passes=1
        )
        
        # 我们的目标 ID 在 segmap 中是 1
        obj_key = 1.0
        if obj_key in grasps_mask and len(grasps_mask[obj_key]) > 0:
            g_msk, s_msk, c_msk = grasps_mask[obj_key], scores_mask[obj_key], contact_pts_mask[obj_key]
            g_msk_nms, s_msk_nms, c_msk_nms = cgn_grasp_nms(g_msk, s_msk, c_msk)
            g_msk_top = g_msk_nms[:50]
            s_msk_top = s_msk_nms[:50]
            c_msk_top = c_msk_nms[:50]
        else:
            g_msk_top, s_msk_top, c_msk_top = np.array([]), np.array([]), np.array([])
            
        # 保存实验结果
        res_mask = {
            "group": "CGN_Mask",
            "num_grasps": len(g_msk_top),
            "top1_score": float(s_msk_top[0]) if len(s_msk_top) > 0 else 0.0,
            "translations": g_msk_top[:, :3, 3].tolist() if len(g_msk_top) > 0 else [],
            "contact_pts": c_msk_top.tolist() if len(c_msk_top) > 0 else []
        }
        mask_json_path = os.path.join(sample_output_dir, "result_group_CGN_Mask.json")
        with open(mask_json_path, 'w') as out_f:
            json.dump(res_mask, out_f, indent=2)
        print(f"   ✓ CGN-Mask 完成，获取 {len(g_msk_top)} 个抓取。已保存至 {mask_json_path}")

    print("\n=== 所有样本 Contact-GraspNet 实验推理已全部完成！ ===")

if __name__ == "__main__":
    main()
