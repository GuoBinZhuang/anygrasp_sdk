#!/bin/bash
# AnyGrasp SDK 部署脚本
# 需要在 anygrasp conda 环境激活后执行，或通过 conda run 调用

set -e  # 出错即退出

WORKSPACE="/home/gb/My_respositories"
ANYGRASP_DIR="$WORKSPACE/anygrasp_sdk"
CUDA_HOME_PATH="/usr/local/cuda-12.4"

echo "=========================================="
echo "AnyGrasp SDK 安装脚本"
echo "=========================================="
echo "工作目录: $WORKSPACE"
echo "CUDA 路径: $CUDA_HOME_PATH"
echo ""

# ---- 步骤 1：检查 anygrasp conda 环境 ----
echo "[步骤 1] 检查 conda 环境..."
if conda env list | grep -q "^anygrasp "; then
    echo "  ✓ anygrasp 环境已存在"
else
    echo "  创建 anygrasp 环境（Python 3.11）..."
    conda create -n anygrasp python=3.11 -y
fi

# ---- 步骤 2：克隆 chenxi-wang 的 MinkowskiEngine（支持 CUDA 12.x）----
echo ""
echo "[步骤 2] 准备 MinkowskiEngine（chenxi-wang 修改版）..."
MINK_DIR="$WORKSPACE/MinkowskiEngine_chenxi"
if [ ! -d "$MINK_DIR" ]; then
    git clone https://github.com/chenxi-wang/MinkowskiEngine.git "$MINK_DIR"
    echo "  ✓ 克隆完成"
else
    echo "  ✓ 已存在，跳过克隆"
fi

# 切换到 CUDA 12.x 支持分支
cd "$MINK_DIR"
git fetch origin
if git branch -a | grep -q "cuda-12-1"; then
    git checkout cuda-12-1
    echo "  ✓ 切换到 cuda-12-1 分支"
else
    echo "  ℹ 未找到 cuda-12-1 分支，使用 master"
fi

# ---- 步骤 3：编译安装 MinkowskiEngine ----
echo ""
echo "[步骤 3] 编译安装 MinkowskiEngine（这需要 5-15 分钟）..."
CONDA_PREFIX=$(conda run -n anygrasp python -c "import sys; print(sys.prefix)" 2>/dev/null)
echo "  CONDA_PREFIX: $CONDA_PREFIX"

conda run -n anygrasp bash -c "
    export CUDA_HOME=$CUDA_HOME_PATH
    export PATH=$CUDA_HOME_PATH/bin:\$PATH
    cd $MINK_DIR
    python setup.py install \
        --blas_include_dirs=$CONDA_PREFIX/include \
        --blas_library_dirs=$CONDA_PREFIX/lib \
        --blas=openblas
"
echo "  ✓ MinkowskiEngine 编译完成"

# ---- 步骤 4：安装 AnyGrasp 其他依赖 ----
echo ""
echo "[步骤 4] 安装 AnyGrasp 依赖..."
cd "$ANYGRASP_DIR"
conda run -n anygrasp pip install -r requirements.txt
echo "  ✓ 依赖安装完成"

# ---- 步骤 5：安装 pointnet2 ----
echo ""
echo "[步骤 5] 编译安装 pointnet2..."
cd "$ANYGRASP_DIR/pointnet2"
conda run -n anygrasp bash -c "
    export CUDA_HOME=$CUDA_HOME_PATH
    export PATH=$CUDA_HOME_PATH/bin:\$PATH
    python setup.py install
"
echo "  ✓ pointnet2 安装完成"

# ---- 步骤 6：设置 gsnet 库 ----
echo ""
echo "[步骤 6] 配置 gsnet 库（Python 3.11 版本）..."
GSNET_DIR="$ANYGRASP_DIR/grasp_detection/gsnet_versions"
TARGET_DIR="$ANYGRASP_DIR/grasp_detection"
cp "$GSNET_DIR/gsnet.cpython-311-x86_64-linux-gnu.so" "$TARGET_DIR/gsnet.so"
# 创建符号链接，Python 导入时用
ln -sf "$TARGET_DIR/gsnet.so" "$TARGET_DIR/gsnet.cpython-311.so" 2>/dev/null || true
echo "  ✓ gsnet.so 已配置"

# ---- 步骤 7：配置 lib_cxx ----
echo ""
echo "[步骤 7] 配置 lib_cxx（License 验证库）..."
LIBCXX_DIR="$ANYGRASP_DIR/license_registration/lib_cxx_versions"
DETECTION_DIR="$ANYGRASP_DIR/grasp_detection"
cp "$LIBCXX_DIR/lib_cxx.cpython-311-x86_64-linux-gnu.so" \
   "$DETECTION_DIR/lib_cxx.cpython-311-x86_64-linux-gnu.so"
echo "  ✓ lib_cxx 已配置到 grasp_detection/"

# ---- 步骤 8：验证安装 ----
echo ""
echo "[步骤 8] 验证安装..."
conda run -n anygrasp python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
import MinkowskiEngine as ME
print(f'  MinkowskiEngine: 导入成功')
" && echo "  ✓ 所有核心依赖验证通过"

echo ""
echo "=========================================="
echo "安装完成！"
echo ""
echo "下一步（手动操作）："
echo "1. 申请 License："
echo "   特征码: $(cd $ANYGRASP_DIR/license_registration && LD_LIBRARY_PATH=/home/gb/miniconda3/envs/anygrasp/lib:$LD_LIBRARY_PATH ./license_checker -f 2>&1)"
echo "   申请地址: https://forms.gle/XVV3Eip8njTYJEBo6"
echo ""
echo "2. 收到 license.zip 后："
echo "   unzip license.zip -d $ANYGRASP_DIR/grasp_detection/license"
echo ""
echo "3. 下载模型权重并运行测试："
echo "   conda run -n anygrasp python grasp_detection/demo.py \\"
echo "     --checkpoint_path grasp_detection/log/checkpoint_detection.tar"
echo "=========================================="
