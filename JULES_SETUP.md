# Jules 快速设置指南

## 📥 第一步：克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/srpsi-engine-tiny.git
cd srpsi-engine-tiny
```

## 🔧 第二步：创建虚拟环境

```bash
# Python 3.10+ required
python -m venv venv
source venv/bin/activate
```

## 📦 第三步：安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} ready')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 第四步：准备训练数据

### 方法 A：从 Mac 复制（推荐）

```bash
# 在 Mac 上运行
scp data/burgers_1d.npy jules@JULES_IP:~/srpsi-engine-tiny/data/
```

### 方法 B：从 Windows 复制

```bash
# 在 Windows 上运行
scp data\burgers_1d.npy jules@JULES_IP:~/srpsi-engine-tiny/data/
```

### 方法 C：生成新数据

```bash
python src/data_gen.py
```

验证数据：
```bash
ls -lh data/burgers_1d.npy
# 应该显示 ~117 MB
```

## 🚀 第五步：开始训练

### 训练 Exp2 (SRΨ Real-only)

```bash
# 后台运行
nohup python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real \
  > logs/ablation_srpsi_real_jules.log 2>&1 &

# 记录进程 ID
echo $! > /tmp/srpsi_real.pid
```

### 训练 Exp3 (SRΨ w/o R) - 可选

```bash
nohup python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_no_r \
  --output outputs/ablation_srpsi_no_r \
  > logs/ablation_srpsi_no_r_jules.log 2>&1 &

echo $! > /tmp/srpsi_no_r.pid
```

## 📊 第六步：监控训练

### 查看实时日志
```bash
tail -f logs/ablation_srpsi_real_jules.log
```

### 检查进度
```bash
# 统计完成的 epoch 数
grep -c "Epoch [0-9]*/80" logs/ablation_srpsi_real_jules.log

# 查看最新 loss
tail -5 logs/ablation_srpsi_real_jules.log | grep loss
```

### 检查 checkpoint
```bash
ls -lh outputs/ablation_srpsi_real/checkpoints/
```

## ⏹️ 停止训练（如果需要）

```bash
# 找到进程 ID
cat /tmp/srpsi_real.pid

# 停止进程
kill $(cat /tmp/srpsi_real.pid)
```

## 📤 第七步：训练完成后

### 找到最佳模型
```bash
ls -lh outputs/*best.pth
```

### 下载到本地 Mac

```bash
# 在 Mac 上运行
scp jules@JULES_IP:~/srpsi-engine-tiny/outputs/ablation_srpsi_real_best.pth \
    outputs/
```

## ⏱️ 预计时间

- 单个实验（CPU）：~15 小时
- 两个实验并行：~15 小时

建议：**只跑 Exp2**，让 Mac/Windows 跑其他实验。

---

**准备就绪！开始训练吧！** 🎯
