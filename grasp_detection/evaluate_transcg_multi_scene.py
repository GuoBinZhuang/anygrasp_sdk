"""
TransCG 多场景批量评估脚本
=========================
扩展 evaluate_transcg_grasp.py，支持跨场景批量评估，
并输出按场景的 per-scene 细粒度统计结果（CSV + JSON）。

用法:
    python evaluate_transcg_multi_scene.py [--scenes 1,5,7,8,11,12] [--max-views 50]

默认使用 TransCG test split 的场景（metadata.json 中定义）。
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from PIL import Image
import torch
import time
from scipy.ndimage import label, binary_dilation, uniform_filter

# 添加 contact_graspnet 路径
sys.path.append('/home/gb/My_respositories/contact_graspnet_pytorch')
sys.path.append('/home/gb/My_respositories/contact_graspnet_pytorch/contact_graspnet_pytorch')

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO


def compute_qd_map(depth: np.ndarray, window_size: int = 5) -> np.ndarray:
    """计算深度质量图 (Qd)"""
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


def cgn_grasp_nms(grasps, scores, contact_pts,
                  translation_threshold=0.03, rotation_threshold_deg=30.0):
    """对 CGN 输出执行 NMS 去重"""
    if len(grasps) == 0:
        return grasps, scores, contact_pts
    idx = np.argsort(-scores)
    grasps, scores, contact_pts = grasps[idx], scores[idx], contact_pts[idx]

    keep, disabled = [], np.zeros(len(grasps), dtype=bool)
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
        cos_ang = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        dist_r = np.degrees(np.arccos(cos_ang))
        conflict = (dist_t < translation_threshold) & (dist_r < rotation_threshold_deg)
        disabled[i+1:][conflict] = True
    return grasps[keep], scores[keep], contact_pts[keep]


def evaluate_viewpoint(grasp_estimator, vp_dir, cam_K, cam_idx, max_grasps=50):
    """
    对单个视角进行 baseline 和 proposed 两种方法的评估。
    返回 dict: {native_prec, native_cnt, proposed_prec, proposed_cnt}
    """
    rgb_path = os.path.join(vp_dir, f"rgb{cam_idx}.png")
    depth_path = os.path.join(vp_dir, f"depth{cam_idx}.png")
    gt_depth_path = os.path.join(vp_dir, f"depth{cam_idx}-gt.png")
    gt_mask_path = os.path.join(vp_dir, f"depth{cam_idx}-gt-mask.png")

    for p in [rgb_path, depth_path, gt_depth_path, gt_mask_path]:
        if not os.path.exists(p):
            return None

    rgb = np.array(Image.open(rgb_path))
    depth_raw = np.array(Image.open(depth_path))      # 16-bit mm
    gt_depth = np.array(Image.open(gt_depth_path))     # 16-bit mm
    gt_mask = np.array(Image.open(gt_mask_path)) > 0
    H, W = depth_raw.shape

    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    depth_raw_m = depth_raw.astype(np.float32) / 1000.0

    def _eval_contacts(contact_pts):
        """将 contact points 投影到 2D，判断是否在 GT mask + 深度对齐"""
        valid_cnt = 0
        for pt in contact_pts:
            x, y, z = pt
            u = int(np.round((x * fx) / z + cx))
            v = int(np.round((y * fy) / z + cy))
            if 0 <= u < W and 0 <= v < H and gt_mask[v, u]:
                z_gt = gt_depth[v, u] / 1000.0
                if z_gt > 0 and abs(z - z_gt) < 0.03:
                    valid_cnt += 1
        return valid_cnt

    obj_key = 1.0

    # ---- 1. Native Baseline ----
    pc_full_nat, pc_seg_nat, _ = grasp_estimator.extract_point_clouds(
        depth_raw_m, cam_K, segmap=gt_mask.astype(np.int32), rgb=rgb, z_range=[0.2, 1.8]
    )
    g_nat, s_nat, c_nat, _ = grasp_estimator.predict_scene_grasps(
        pc_full_nat, pc_segments=pc_seg_nat, local_regions=True,
        filter_grasps=True, forward_passes=1
    )
    if obj_key in g_nat and len(g_nat[obj_key]) > 0:
        _g, _s, _c = cgn_grasp_nms(g_nat[obj_key], s_nat[obj_key], c_nat[obj_key])
        _g, _s, _c = _g[:max_grasps], _s[:max_grasps], _c[:max_grasps]
    else:
        _c = []
    native_valid = _eval_contacts(_c) if len(_c) > 0 else 0
    native_prec = native_valid / len(_c) if len(_c) > 0 else 0.0

    # ---- 2. Proposed Pipeline ----
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
        labeled_arr, num_features = label(valid_mask_isolated)
        if num_features > 0:
            bincounts = np.bincount(labeled_arr.ravel())
            largest_label = np.argmax(bincounts[1:]) + 1
            largest_cc = (labeled_arr == largest_label)
            valid_mask = binary_dilation(largest_cc, structure=np.ones((5, 5)))

    depth_proc_m = depth_raw_m.copy()
    depth_proc_m[~valid_mask] = 0.0

    pc_full_prop, pc_seg_prop, _ = grasp_estimator.extract_point_clouds(
        depth_proc_m, cam_K, segmap=valid_mask.astype(np.int32), rgb=rgb, z_range=[0.2, 1.8]
    )
    g_prop, s_prop, c_prop, _ = grasp_estimator.predict_scene_grasps(
        pc_full_prop, pc_segments=pc_seg_prop, local_regions=True,
        filter_grasps=True, forward_passes=1
    )
    if obj_key in g_prop and len(g_prop[obj_key]) > 0:
        _g2, _s2, _c2 = cgn_grasp_nms(g_prop[obj_key], s_prop[obj_key], c_prop[obj_key])
        _g2, _s2, _c2 = _g2[:max_grasps], _s2[:max_grasps], _c2[:max_grasps]
    else:
        _c2 = []
    proposed_valid = _eval_contacts(_c2) if len(_c2) > 0 else 0
    proposed_prec = proposed_valid / len(_c2) if len(_c2) > 0 else 0.0

    return {
        "native_prec": native_prec,
        "native_cnt": len(_c),
        "proposed_prec": proposed_prec,
        "proposed_cnt": len(_c2),
    }


def main():
    parser = argparse.ArgumentParser(description="TransCG 多场景评估")
    parser.add_argument("--dataset-dir", type=str,
                        default="/mnt/data/gb/graspnet/transcg_extracted",
                        help="TransCG 根目录")
    parser.add_argument("--scenes", type=str, default="",
                        help="逗号分隔的场景编号，为空则使用 test split")
    parser.add_argument("--max-views", type=int, default=0,
                        help="每个场景最多评估的视角数（0 = 全部）")
    parser.add_argument("--output-dir", type=str,
                        default="/home/gb/My_respositories/anygrasp_sdk/grasp_detection",
                        help="结果输出目录")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # 确定要评估的场景列表
    if args.scenes:
        scene_ids = [int(x) for x in args.scenes.split(",")]
    else:
        # 使用 metadata.json 中的 test split
        meta_path = os.path.join(dataset_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            scene_ids = sorted(meta.get("test", []))
        else:
            # 回退：自动发现所有场景
            scene_ids = sorted([
                int(d.replace("scene", ""))
                for d in os.listdir(dataset_dir)
                if d.startswith("scene") and os.path.isdir(os.path.join(dataset_dir, d))
            ])

    # 只保留实际存在的场景
    available_scene_ids = []
    for sid in scene_ids:
        scene_path = os.path.join(dataset_dir, f"scene{sid}")
        if os.path.isdir(scene_path):
            available_scene_ids.append(sid)
    scene_ids = available_scene_ids

    print(f"=== TransCG 多场景评估 ===")
    print(f"数据集目录: {dataset_dir}")
    print(f"待评估场景数: {len(scene_ids)}")
    print(f"场景列表: {scene_ids}")
    print()

    # 载入 CGN 模型
    ckpt_dir = '/home/gb/My_respositories/contact_graspnet_pytorch/checkpoints/contact_graspnet'
    global_config = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=[])
    grasp_estimator = GraspEstimator(global_config)
    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir,
                                 model=grasp_estimator.model)
    checkpoint_io.load('model.pt')
    print("✓ Contact-GraspNet 权重加载成功！\n")

    # 载入相机内参
    intrinsics_dir = os.path.join(dataset_dir, "camera_intrinsics")
    cam_K_d435 = np.load(os.path.join(intrinsics_dir, "1-camIntrinsics-D435.npy"))
    cam_K_l515 = np.load(os.path.join(intrinsics_dir, "2-camIntrinsics-L515.npy"))

    # 按场景遍历
    all_results = []          # 全局视角级别结果
    per_scene_summary = []    # 按场景汇总

    global_t_start = time.time()
    global_processed = 0

    for scene_idx, sid in enumerate(scene_ids):
        scene_dir = os.path.join(dataset_dir, f"scene{sid}")
        viewpoints = sorted([d for d in os.listdir(scene_dir) if d.isdigit()], key=int)
        if args.max_views > 0:
            viewpoints = viewpoints[:args.max_views]

        scene_native_precs, scene_native_cnts = [], []
        scene_proposed_precs, scene_proposed_cnts = [], []
        scene_processed = 0
        scene_t_start = time.time()

        print(f"[Scene {sid}] ({scene_idx+1}/{len(scene_ids)}) "
              f"共 {len(viewpoints)} 个视角...")

        for vp in viewpoints:
            vp_dir = os.path.join(scene_dir, vp)
            # 自适应选择相机
            if os.path.exists(os.path.join(vp_dir, "rgb1.png")):
                cam_idx, cam_K = "1", cam_K_d435
            elif os.path.exists(os.path.join(vp_dir, "rgb2.png")):
                cam_idx, cam_K = "2", cam_K_l515
            else:
                continue

            try:
                result = evaluate_viewpoint(grasp_estimator, vp_dir, cam_K, cam_idx)
                if result is None:
                    continue

                scene_native_precs.append(result["native_prec"])
                scene_native_cnts.append(result["native_cnt"])
                scene_proposed_precs.append(result["proposed_prec"])
                scene_proposed_cnts.append(result["proposed_cnt"])

                all_results.append({
                    "scene": sid,
                    "viewpoint": int(vp),
                    **result,
                })
                scene_processed += 1
                global_processed += 1

                if scene_processed % 20 == 0:
                    elapsed = time.time() - scene_t_start
                    print(f"  [{scene_processed}/{len(viewpoints)}] 耗时 {elapsed:.1f}s")

            except Exception as e:
                print(f"  ⚠ scene{sid}/vp{vp} 评估失败: {e}")
                continue

        # 场景级汇总
        if len(scene_native_precs) > 0:
            summary = {
                "scene": sid,
                "n_views": len(scene_native_precs),
                "native_prec_mean": float(np.mean(scene_native_precs)),
                "native_prec_std": float(np.std(scene_native_precs)),
                "native_cnt_mean": float(np.mean(scene_native_cnts)),
                "proposed_prec_mean": float(np.mean(scene_proposed_precs)),
                "proposed_prec_std": float(np.std(scene_proposed_precs)),
                "proposed_cnt_mean": float(np.mean(scene_proposed_cnts)),
            }
            per_scene_summary.append(summary)

            scene_elapsed = time.time() - scene_t_start
            print(f"  ✓ Scene {sid} 完成: "
                  f"N={summary['n_views']}, "
                  f"Baseline={summary['native_prec_mean']:.4f}±{summary['native_prec_std']:.4f} "
                  f"(cnt={summary['native_cnt_mean']:.1f}), "
                  f"Proposed={summary['proposed_prec_mean']:.4f}±{summary['proposed_prec_std']:.4f} "
                  f"(cnt={summary['proposed_cnt_mean']:.1f}), "
                  f"用时 {scene_elapsed:.1f}s")
        else:
            print(f"  ❌ Scene {sid} 无有效视角")

    # ==== 全局汇总 ====
    total_elapsed = time.time() - global_t_start
    print(f"\n{'='*70}")
    print(f"全局评估完成: {global_processed} 个视角, "
          f"{len(per_scene_summary)} 个场景, 总耗时 {total_elapsed:.1f}s")
    print(f"{'='*70}")

    if len(all_results) > 0:
        all_native_precs = [r["native_prec"] for r in all_results]
        all_native_cnts = [r["native_cnt"] for r in all_results]
        all_proposed_precs = [r["proposed_prec"] for r in all_results]
        all_proposed_cnts = [r["proposed_cnt"] for r in all_results]

        print(f"\n总视角数 N = {len(all_results)}")
        print(f"{'Method':<35} | {'3D Precision':>20} | {'Avg Candidates':>15}")
        print("-" * 75)
        print(f"{'Baseline (CGN-Mask on Raw)':<35} | "
              f"{np.mean(all_native_precs):.4f} ± {np.std(all_native_precs):.4f} | "
              f"{np.mean(all_native_cnts):.1f}")
        print(f"{'Proposed (Qd+Jump+LCC)':<35} | "
              f"{np.mean(all_proposed_precs):.4f} ± {np.std(all_proposed_precs):.4f} | "
              f"{np.mean(all_proposed_cnts):.1f}")
        print("=" * 75)

        # ==== 保存结果 ====
        # 1. 视角级别 CSV
        csv_path = os.path.join(output_dir, "transcg_multi_scene_per_view.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scene", "viewpoint",
                "native_prec", "native_cnt",
                "proposed_prec", "proposed_cnt"
            ])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ 视角级别结果已保存至: {csv_path}")

        # 2. 场景级别 JSON
        json_path = os.path.join(output_dir, "transcg_multi_scene_summary.json")
        summary_output = {
            "total_views": len(all_results),
            "total_scenes": len(per_scene_summary),
            "elapsed_seconds": total_elapsed,
            "global_stats": {
                "baseline": {
                    "precision_mean": float(np.mean(all_native_precs)),
                    "precision_std": float(np.std(all_native_precs)),
                    "candidates_mean": float(np.mean(all_native_cnts)),
                },
                "proposed": {
                    "precision_mean": float(np.mean(all_proposed_precs)),
                    "precision_std": float(np.std(all_proposed_precs)),
                    "candidates_mean": float(np.mean(all_proposed_cnts)),
                },
            },
            "per_scene": per_scene_summary,
        }
        with open(json_path, "w") as f:
            json.dump(summary_output, f, indent=2)
        print(f"✓ 场景级别汇总已保存至: {json_path}")

        # 3. 场景级别 CSV
        scene_csv_path = os.path.join(output_dir, "transcg_multi_scene_per_scene.csv")
        with open(scene_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scene", "n_views",
                "native_prec_mean", "native_prec_std", "native_cnt_mean",
                "proposed_prec_mean", "proposed_prec_std", "proposed_cnt_mean"
            ])
            writer.writeheader()
            writer.writerows(per_scene_summary)
        print(f"✓ 场景级别 CSV 已保存至: {scene_csv_path}")
    else:
        print("❌ 未成功评估任何有效视角")


if __name__ == "__main__":
    main()
