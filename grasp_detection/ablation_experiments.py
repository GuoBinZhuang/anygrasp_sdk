"""
消融实验框架：基于深度质量图的 SAM2 辅助半透明袋装物体抓取区域检测
四组对比实验：
  Group A: RGB-only → AnyGrasp（基线）
  Group B: RGB-D → AnyGrasp（原始深度，验证深度噪声问题）
  Group C: RGB-D + Qd 过滤 → AnyGrasp（验证 Qd 有效性）
  Group D: RGB-D + Qd + SAM2 提示 → AnyGrasp（完整方法）
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
import open3d as o3d
import json
import time

AnyGrasp = None
from graspnetAPI import GraspGroup
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  [Seed] Global seed set to {seed}")

# ====================== 参数配置 ======================
parser = argparse.ArgumentParser(description='AnyGrasp 消融实验')
parser.add_argument('--checkpoint_path', required=True, help='模型检查点路径')
parser.add_argument('--data_dir', default='./example_data', help='数据目录')
parser.add_argument('--group', type=str, choices=['A', 'B', 'C', 'D', 'all'],
                    default='all', help='运行哪组实验: A/B/C/D/all')
parser.add_argument('--qd_ir_path', default=None, help='IR 强度图路径（Group C/D 需要）')
parser.add_argument('--sam2_mask_path', default=None, help='SAM2 分割 mask 路径（Group D 需要）')
parser.add_argument('--output_dir', default='./ablation_results', help='结果输出目录')
parser.add_argument('--max_gripper_width', type=float, default=0.1)
parser.add_argument('--gripper_height', type=float, default=0.03)
parser.add_argument('--top_down_grasp', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_graspnet_baseline', action='store_true', help='使用 GraspNet-baseline 运行')
parser.add_argument('--num_candidates', type=int, default=50, help='各组固定截取前K个候选抓取评估')
parser.add_argument('--lims', type=float, nargs=6, default=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.2],
                    help='xmin xmax ymin ymax zmin zmax 工作空间边界')
parser.add_argument('--dense_grasp', action='store_true', help='各组统一的密集抓取采样模式')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


def build_point_cloud(depth: np.ndarray, color: np.ndarray,
                       fx: float, fy: float, cx: float, cy: float,
                       scale: float = 1000.0, depth_mask: np.ndarray = None):
    """
    将深度图和彩色图转换为点云。
    
    Args:
        depth:      HxW 深度图（原始整数值）
        color:      HxWx3 彩色图（float32, 0-1）
        fx, fy:     焦距
        cx, cy:     主点
        scale:      深度缩放因子（通常 1000 表示毫米→米）
        depth_mask: 可选的额外过滤 mask（True 表示保留），用于 Qd 或 SAM2 过滤
    
    Returns:
        points: Nx3 点云坐标
        colors: Nx3 颜色
        valid_mask: HxW bool，表示哪些像素被保留
    """
    xmap, ymap = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    points_z = depth / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # 基础深度有效性过滤（z > 0 且 z < 1m）
    valid = (points_z > 0) & (points_z < 1)

    # 叠加外部 mask（Qd 过滤 / SAM2 mask）
    if depth_mask is not None:
        valid = valid & depth_mask

    points = np.stack([points_x, points_y, points_z], axis=-1)
    pts_valid = points[valid].astype(np.float32)
    clr_valid = color[valid].astype(np.float32)

    return pts_valid, clr_valid, valid


def compute_qd_map(depth: np.ndarray, ir_intensity: np.ndarray = None,
                    window_size: int = 5) -> np.ndarray:
    """
    构建深度质量图 Qd。
    
    融合策略：
      1. 局部深度统计：计算滑动窗口内的深度标准差（std），
         std 大的区域说明深度不可靠（袋子边缘/半透明区域噪声大）
      2. IR 强度（可选）：IR 强度低的区域深度通常不可靠
    
    Args:
        depth:        HxW 深度图（原始整数值）
        ir_intensity: HxW IR 强度图（可选，uint8 或 uint16）
        window_size:  局部统计窗口大小
    
    Returns:
        qd: HxW float32 质量图，值域 [0, 1]，越高越可靠
    """
    from scipy.ndimage import uniform_filter, generic_filter

    depth_f = depth.astype(np.float32)

    # --- 深度统计项：局部标准差 ---
    # 有效深度掩码
    valid = (depth > 0).astype(np.float32)
    
    # 局部均值（只统计有效像素）
    local_sum = uniform_filter(depth_f * valid, size=window_size)
    local_cnt = uniform_filter(valid, size=window_size) + 1e-6
    local_mean = local_sum / local_cnt

    # 局部方差 E[X^2] - (E[X])^2
    local_sq_sum = uniform_filter((depth_f ** 2) * valid, size=window_size)
    local_sq_mean = local_sq_sum / local_cnt
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # 归一化标准差，转化为质量分数（std 低 → 质量高）
    std_max = np.percentile(local_std[valid > 0], 95) + 1e-6
    q_depth_stat = 1.0 - np.clip(local_std / std_max, 0, 1)

    # --- IR 强度项（可选）---
    if ir_intensity is not None:
        ir_f = ir_intensity.astype(np.float32)
        ir_max = np.percentile(ir_f, 99) + 1e-6
        q_ir = np.clip(ir_f / ir_max, 0, 1)
        # 融合权重：深度统计 0.6，IR 强度 0.4
        qd = 0.6 * q_depth_stat + 0.4 * q_ir
    else:
        qd = q_depth_stat

    # 无效深度区域 qd = 0
    qd[depth == 0] = 0.0
    return qd.astype(np.float32)


def load_sam2_mask(mask_path: str, image_shape: tuple) -> np.ndarray:
    """
    加载 SAM2 生成的分割 mask。
    
    支持格式：
      - PNG 图像（白色=目标区域）
      - npy 数组（bool HxW）
    
    Returns:
        mask: HxW bool，True 表示目标像素
    """
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path).astype(bool)
    else:
        mask_img = np.array(Image.open(mask_path).convert('L'))
        mask = mask_img > 127
    
    # 确保和图像尺寸一致
    if mask.shape != image_shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((image_shape[1], image_shape[0]),
                                   PILImage.NEAREST)
        mask = np.array(mask_pil) > 127

    return mask


class GraspNetBaselineWrapper:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = None

    def load_net(self):
        import sys
        sys.path.append('/home/gb/My_respositories/graspnet-baseline')
        sys.path.append('/home/gb/My_respositories/graspnet-baseline/models')
        sys.path.append('/home/gb/My_respositories/graspnet-baseline/pointnet2')
        sys.path.append('/home/gb/My_respositories/graspnet-baseline/utils')

        from models.graspnet import GraspNet
        self.net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, is_training=False)
        self.net.to(self.device)

        print(f"  [GraspNetWrapper] 加载模型权重：{self.cfgs.checkpoint_path}")
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        print("  [GraspNetWrapper] ✓ 模型和权重加载成功！")

    def get_grasp(self, points, colors, lims=None, apply_object_mask=True, dense_grasp=False, collision_detection=True):
        from models.graspnet import pred_decode
        from graspnetAPI import GraspGroup
        import open3d as o3d

        # 1. 过滤 lims 范围内的点云
        if lims is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = lims
            mask = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & \
                   (points[:, 1] >= ymin) & (points[:, 1] <= ymax) & \
                   (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
            points = points[mask]
            colors = colors[mask]

        if len(points) == 0:
            return GraspGroup(), o3d.geometry.PointCloud()

        # 2. 构造 open3d.geometry.PointCloud 用于返回
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

        # 3. 准备送入模型的输入数据 (下采样到 20000)
        num_points = 20000
        if len(points) >= num_points:
            idxs = np.random.choice(len(points), num_points, replace=False)
        else:
            idxs = np.random.choice(len(points), num_points, replace=True)
        pts_input = points[idxs]
        clrs_input = colors[idxs]

        end_points = {}
        end_points['point_clouds'] = torch.from_numpy(pts_input).unsqueeze(0).to(self.device)
        end_points['cloud_colors'] = torch.from_numpy(clrs_input).unsqueeze(0).to(self.device)

        # 4. 前向传递
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
            gg_array = grasp_preds[0].detach().cpu().numpy()
            gg = GraspGroup(gg_array)

        # 5. 过滤掉不符合 lims 的抓取
        if lims is not None:
            gg_translations = gg.translations
            xmin, xmax, ymin, ymax, zmin, zmax = lims
            in_lims = (gg_translations[:, 0] >= xmin) & (gg_translations[:, 0] <= xmax) & \
                      (gg_translations[:, 1] >= ymin) & (gg_translations[:, 1] <= ymax) & \
                      (gg_translations[:, 2] >= zmin) & (gg_translations[:, 2] <= zmax)
            filtered_array = gg_array[in_lims]
            gg = GraspGroup(filtered_array)

        return gg, cloud


def grasp_nms(gg, translation_threshold=0.03, rotation_threshold=30.0):
    """
    纯 Python/numpy 实现的 GraspGroup NMS。
    """
    if len(gg) == 0:
        return gg
    
    # 按照得分从高到低排序
    gg = gg.sort_by_score()
    gg_array = gg.grasp_group_array
    
    translations = gg.translations  # (N, 3)
    rotations = gg.rotation_matrices  # (N, 3, 3)
    approaches = rotations[:, :, 0]  # (N, 3) 用 approach 向量作为抓取朝向
    
    keep = []
    num_grasps = len(gg)
    disabled = np.zeros(num_grasps, dtype=bool)
    
    for i in range(num_grasps):
        if disabled[i]:
            continue
        keep.append(i)
        
        # 计算后续抓取与当前抓取的平移距离
        dist_t = np.linalg.norm(translations[i:i+1] - translations[i+1:], axis=1)
        
        # 计算后续抓取与当前抓取的角度差异 (点积即 cos theta)
        cos_ang = np.sum(approaches[i:i+1] * approaches[i+1:], axis=1)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        dist_r = np.degrees(np.arccos(cos_ang))
        
        # 如果距离和角度均小于阈值，则标记为失效
        conflict = (dist_t < translation_threshold) & (dist_r < rotation_threshold)
        
        # 更新 disabled 数组（注意索引偏移）
        disabled[i+1:][conflict] = True
        
    filtered_array = gg_array[keep]
    from graspnetAPI import GraspGroup
    return GraspGroup(filtered_array)


def run_anygrasp(anygrasp, points: np.ndarray, colors: np.ndarray,
                 lims: list, apply_object_mask: bool = True,
                 dense_grasp: bool = False) -> GraspGroup:
    """
    调用 AnyGrasp 推理并返回抓取姿态列表。
    """
    if len(points) == 0:
        print("  ⚠ 点云为空，跳过推理")
        return GraspGroup()

    gg, cloud = anygrasp.get_grasp(
        points, colors, lims=lims,
        apply_object_mask=apply_object_mask,
        dense_grasp=dense_grasp,
        collision_detection=True
    )

    if len(gg) == 0:
        print("  ⚠ 碰撞检测后无抓取点")
        return GraspGroup()

    gg = grasp_nms(gg).sort_by_score()
    if len(gg) > cfgs.num_candidates:
        gg = gg[:cfgs.num_candidates]
    return gg


def save_results(group_name: str, gg: GraspGroup, elapsed: float,
                 output_dir: str, cloud: o3d.geometry.PointCloud = None):
    """
    保存实验结果（JSON + 可视化点云）。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scores = gg.scores if len(gg) > 0 else []
    top5_avg = float(np.mean(scores[:5])) if len(scores) >= 5 else (float(np.mean(scores)) if len(scores) > 0 else 0.0)
    top10_avg = float(np.mean(scores[:10])) if len(scores) >= 10 else (float(np.mean(scores)) if len(scores) > 0 else 0.0)
    
    result = {
        'group': group_name,
        'num_grasps': len(gg),
        'top1_score': float(gg[0].score) if len(gg) > 0 else 0.0,
        'top5_avg_score': top5_avg,
        'top10_avg_score': top10_avg,
        'top5_scores': [float(s) for s in scores[:5]],
        'inference_time_s': elapsed,
        'translations': gg.translations.tolist() if len(gg) > 0 else []
    }

    json_path = os.path.join(output_dir, f'result_group_{group_name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ 结果保存到 {json_path}")
    print(f"  ✓ 检测到 {len(gg)} 个抓取，Top-1 得分: {result['top1_score']:.4f}，Top-5 均值: {result['top5_avg_score']:.4f}")
    return result


def load_intrinsics(data_dir):
    """
    自适应从数据目录中加载相机内参
    """
    intrinsics_path = os.path.join(data_dir, 'intrinsics.txt')
    if os.path.exists(intrinsics_path):
        try:
            with open(intrinsics_path, 'r') as f:
                line = f.readline().strip()
                parts = [float(x) for x in line.split()]
                if len(parts) >= 5:
                    fx, fy, cx, cy, scale = parts[0], parts[1], parts[2], parts[3], parts[4]
                    print(f"  ✓ 从 {intrinsics_path} 加载内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}, scale={scale}")
                    return fx, fy, cx, cy, scale
        except Exception as e:
            print(f"  ⚠ 读取内参文件失败，使用默认值: {e}")
    # 默认 anygrasp 原始 example 内参
    return 927.17, 927.37, 651.32, 349.62, 1000.0


def experiment_group_a(anygrasp, data_dir, lims, output_dir):
    """
    Group A：RGB-only → AnyGrasp（基线）
    方法：将深度图置为常数（固定平面深度），相当于忽略真实深度信息。
    目的：建立 RGB-only 的最弱基线，证明深度信息的必要性。
    """
    print("\n[Group A] RGB-only → AnyGrasp 基线")
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    
    # 自动加载或默认相机内参
    fx, fy, cx, cy, scale = load_intrinsics(data_dir)
    
    # 关键操作：用有效深度的中位数填充所有像素（模拟 RGB-only 假设）
    valid_depths = depth[depth > 0]
    if len(valid_depths) > 0:
        median_depth = np.median(valid_depths).astype(depth.dtype)
    else:
        median_depth = 500  # 默认 0.5m
    
    depth_rgb_only = np.where(depth > 0, median_depth, 0)
    
    t0 = time.time()
    points, clrs, _ = build_point_cloud(depth_rgb_only, colors, fx, fy, cx, cy, scale)
    gg = run_anygrasp(anygrasp, points, clrs, lims, apply_object_mask=False, dense_grasp=cfgs.dense_grasp)
    elapsed = time.time() - t0
    
    return save_results('A', gg, elapsed, output_dir)


def experiment_group_b(anygrasp, data_dir, lims, output_dir):
    """
    Group B：RGB-D → AnyGrasp（原始深度）
    方法：直接使用原始深度图，不做任何过滤。
    目的：证明半透明袋子上的深度噪声导致抓取质量下降。
    """
    print("\n[Group B] RGB-D → AnyGrasp（原始深度）")
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    fx, fy, cx, cy, scale = load_intrinsics(data_dir)

    t0 = time.time()
    points, clrs, _ = build_point_cloud(depth, colors, fx, fy, cx, cy, scale)
    gg = run_anygrasp(anygrasp, points, clrs, lims, apply_object_mask=True, dense_grasp=cfgs.dense_grasp)
    elapsed = time.time() - t0

    result = save_results('B', gg, elapsed, output_dir)
    result['num_points'] = len(points)
    return result


def experiment_group_c(anygrasp, data_dir, lims, output_dir,
                        ir_path=None, qd_threshold=0.4):
    """
    Group C：RGB-D + Qd 过滤 → AnyGrasp
    方法：用 Qd 图过滤低质量深度点，保留高质量区域的点云。
    目的：证明 Qd 过滤的有效性（对比 Group B 应有提升）。
    
    Args:
        qd_threshold: Qd 低于此阈值的像素被过滤，论文中建议消融此参数。
    """
    print(f"\n[Group C] RGB-D + Qd 过滤 → AnyGrasp（阈值={qd_threshold}）")
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    fx, fy, cx, cy, scale = load_intrinsics(data_dir)

    # 加载 IR 图（可选）
    ir = None
    if ir_path and os.path.exists(ir_path):
        ir = np.array(Image.open(ir_path))
        print(f"  ✓ 加载 IR 图：{ir_path}")
    else:
        print("  ℹ 未提供 IR 图，仅使用深度统计构建 Qd")

    # 构建 Qd 图
    qd = compute_qd_map(depth, ir_intensity=ir)
    
    # 保存 Qd 图以便论文可视化
    qd_vis = (qd * 255).astype(np.uint8)
    qd_save_path = os.path.join(output_dir, 'qd_map.png')
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(qd_vis).save(qd_save_path)
    print(f"  ✓ Qd 图保存到 {qd_save_path}")
    
    # Qd 过滤 mask
    depth_mask = qd >= qd_threshold

    t0 = time.time()
    points, clrs, _ = build_point_cloud(depth, colors, fx, fy, cx, cy, scale,
                                         depth_mask=depth_mask)
    gg = run_anygrasp(anygrasp, points, clrs, lims, apply_object_mask=True, dense_grasp=cfgs.dense_grasp)
    elapsed = time.time() - t0

    result = save_results('C', gg, elapsed, output_dir)
    result['num_points'] = len(points)
    result['qd_threshold'] = qd_threshold
    result['points_retained_ratio'] = float(depth_mask.mean())
    print(f"  ✓ Qd 过滤后保留 point 数比例: {result['points_retained_ratio']:.2%}")
    return result


def experiment_group_d(anygrasp, data_dir, lims, output_dir,
                        ir_path=None, sam2_mask_path=None, qd_threshold=0.4):
    """
    Group D：RGB-D + Qd + SAM2 提示 → AnyGrasp（完整方法）
    方法：
      1. 用 Qd 图生成 SAM2 提示点（高 Qd 区域内的正提示，袋子边缘作为负提示）
      2. SAM2 输出袋装物体的精确分割 mask
      3. 以 mask 裁剪点云后，用 dense_grasp=True 进行密集抓取检测
    
    注：如果已有预计算的 SAM2 mask，直接加载；否则跳过（需要 SAM2 环境）
    """
    print("\n[Group D] RGB-D + Qd + SAM2 → AnyGrasp（完整方法）")
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    fx, fy, cx, cy, scale = load_intrinsics(data_dir)

    # --- 步骤 1：构建 Qd 图 ---
    ir = None
    if ir_path and os.path.exists(ir_path):
        ir = np.array(Image.open(ir_path))

    qd = compute_qd_map(depth, ir_intensity=ir)
    
    # --- 步骤 2：加载或生成 SAM2 mask ---
    if sam2_mask_path and os.path.exists(sam2_mask_path):
        # 直接加载预计算 mask（推荐：先在 oas-bagseg 环境跑 SAM2）
        sam2_mask = load_sam2_mask(sam2_mask_path, depth.shape)
        print(f"  ✓ 加载 SAM2 mask：{sam2_mask_path}")
    else:
        # 没有预计算 mask 时，用 Qd 阈值 + 形态学操作近似
        print("  ℹ 未提供 SAM2 mask，用 Qd 阈值粗分割代替（仅用于快速测试）")
        from scipy.ndimage import binary_dilation, binary_fill_holes
        sam2_mask = qd >= qd_threshold
        sam2_mask = binary_fill_holes(sam2_mask)
        sam2_mask = binary_dilation(sam2_mask, iterations=3)

    # 保存原始的 SAM2 mask 以供后续目标命中率分析
    np.save(os.path.join(output_dir, 'sam2_mask.npy'), sam2_mask)
    print(f"  ✓ 保存 SAM2 mask 到 {output_dir}")

    # --- 步骤 3：mask 裁剪点云 ---
    # 同时叠加 Qd 过滤（Qd 引导的精细版）
    depth_mask = sam2_mask & (qd >= qd_threshold * 0.7)  # mask 内部可放宽 Qd 阈值

    t0 = time.time()
    points, clrs, _ = build_point_cloud(depth, colors, fx, fy, cx, cy, scale,
                                         depth_mask=depth_mask)
    
    # 关键区别：各组统一采样参数 dense_grasp=cfgs.dense_grasp
    gg = run_anygrasp(anygrasp, points, clrs, lims, apply_object_mask=True, dense_grasp=cfgs.dense_grasp)
    elapsed = time.time() - t0

    # 保存最终三维裁剪掩码以供后续实证反查误杀
    np.save(os.path.join(output_dir, 'depth_mask.npy'), depth_mask)
    print(f"  ✓ 保存 Group D depth_mask.npy 到 {output_dir}")

    result = save_results('D', gg, elapsed, output_dir)

    # === 对照实验：D_expand 掩码向外膨胀 20 像素 (Context Loss 验证) ===
    print("  [Group D_expand] 开启 SAM2 mask 膨胀 20 像素对比实验...")
    from scipy.ndimage import binary_dilation
    # 使用 20 像素的膨胀（iterations=20）
    sam2_mask_expand = binary_dilation(sam2_mask, iterations=20)
    depth_mask_expand = sam2_mask_expand & (qd >= qd_threshold * 0.7)
    
    t0_exp = time.time()
    points_exp, clrs_exp, _ = build_point_cloud(depth, colors, fx, fy, cx, cy, scale,
                                                 depth_mask=depth_mask_expand)
    
    # 强制在相同种子状态下重新推理，保证下采样点云一致
    set_seed(42)
    gg_exp = run_anygrasp(anygrasp, points_exp, clrs_exp, lims, apply_object_mask=True, dense_grasp=cfgs.dense_grasp)
    elapsed_exp = time.time() - t0_exp
    
    save_results('D_expand', gg_exp, elapsed_exp, output_dir)

    result['qd_threshold'] = qd_threshold
    result['sam2_mask_used'] = sam2_mask_path is not None
    result['points_retained_ratio'] = float(depth_mask.mean())
    print(f"  ✓ 完整方法保留点数比例: {result['points_retained_ratio']:.2%}")
    return result


def print_comparison_table(results: list):
    """
    打印四组实验对比表，方便论文数据填写。
    """
    print("\n" + "=" * 80)
    print("消融实验对比结果")
    print("=" * 80)
    print(f"{'组别':<8} {'方法':<30} {'抓取数':<10} {'Top-1分数':<12} {'Top-5均值':<12} {'耗时(s)':<10}")
    print("-" * 80)
    
    group_names = {
        'A': 'RGB-only → AnyGrasp',
        'B': 'RGB-D → AnyGrasp',
        'C': 'RGB-D + Qd 过滤 → AnyGrasp',
        'D': 'RGB-D + Qd + SAM2 → AnyGrasp',
    }
    
    for r in results:
        g = r.get('group', '?')
        print(f"Group {g:<4} {group_names.get(g, ''):<30} "
              f"{r['num_grasps']:<10} {r['top1_score']:<12.4f} {r.get('top5_avg_score', 0.0):<12.4f} {r['inference_time_s']:<10.2f}")
    print("=" * 80)


def main():
    set_seed(42)
    global AnyGrasp
    # 初始化检测器（支持 AnyGrasp / GraspNet-baseline）
    if not cfgs.use_graspnet_baseline:
        try:
            from gsnet import AnyGrasp
            anygrasp = AnyGrasp(cfgs)
            anygrasp.load_net()
        except Exception as e:
            print(f"⚠ 无法加载 AnyGrasp 模块（通常由于缺少授权证书），将强制使用 GraspNet-baseline。错误: {e}")
            cfgs.use_graspnet_baseline = True

    if cfgs.use_graspnet_baseline:
        print("  [Main] 将使用 GraspNet-baseline 运行消融实验...")
        anygrasp = GraspNetBaselineWrapper(cfgs)
        anygrasp.load_net()

    # 工作空间边界（自适应命令行参数）
    lims = cfgs.lims

    os.makedirs(cfgs.output_dir, exist_ok=True)
    results = []

    groups_to_run = ['A', 'B', 'C', 'D'] if cfgs.group == 'all' else [cfgs.group]

    if 'A' in groups_to_run:
        results.append(experiment_group_a(anygrasp, cfgs.data_dir, lims, cfgs.output_dir))

    if 'B' in groups_to_run:
        results.append(experiment_group_b(anygrasp, cfgs.data_dir, lims, cfgs.output_dir))

    if 'C' in groups_to_run:
        results.append(experiment_group_c(
            anygrasp, cfgs.data_dir, lims, cfgs.output_dir,
            ir_path=cfgs.qd_ir_path
        ))

    if 'D' in groups_to_run:
        results.append(experiment_group_d(
            anygrasp, cfgs.data_dir, lims, cfgs.output_dir,
            ir_path=cfgs.qd_ir_path,
            sam2_mask_path=cfgs.sam2_mask_path
        ))

    # 汇总输出
    # 增加 Bug 排查 assertion：若 B 组和 C 组都跑了，它们的点云大小必须不同（C 组有 Qd 过滤，点云点数应当少于 B 组）
    run_groups = [r['group'] for r in results]
    if 'B' in run_groups and 'C' in run_groups:
        num_pts_B = next(r['num_points'] for r in results if r['group'] == 'B')
        num_pts_C = next(r['num_points'] for r in results if r['group'] == 'C')
        assert num_pts_C < num_pts_B, f"Error: Group B and Group C share the exact same point cloud size ({num_pts_B} vs {num_pts_C})!"
        print(f"  ✓ Bug Check Assertion Passed: Group B and Group C point cloud sizes are different ({num_pts_B} vs {num_pts_C}).")

    print_comparison_table(results)

    # 保存汇总 JSON
    summary_path = os.path.join(cfgs.output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 消融实验汇总保存到 {summary_path}")


if __name__ == '__main__':
    main()
