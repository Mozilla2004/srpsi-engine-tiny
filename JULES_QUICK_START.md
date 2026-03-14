# Jules 快速启动指南

## 🎯 你的任务

训练 **Exp2 (SRΨ Real-only)** 模型，验证复值表示的重要性。

## 📋 方案选择

### 方案 A：自动化脚本（推荐）

```bash
# 进入项目目录
cd ~/srpsi-engine-tiny

# 运行自动化脚本
python jules_train_helper.py
```

这个脚本会自动：
- ✅ 检查/创建虚拟环境
- ✅ 检查/安装依赖
- ✅ 检查/生成训练数据
- ✅ 创建必要目录
- ✅ 启动训练
- ✅ 显示监控命令

### 方案 B：手动执行（完全控制）

```bash
cd ~/srpsi-engine-tiny

# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 3. 生成数据（如果没有）
mkdir -p data
python src/data_gen.py

# 4. 创建日志目录
mkdir -p logs

# 5. 启动训练
nohup python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real \
  > logs/ablation_srpsi_real_jules.log 2>&1 &

# 6. 记录进程 ID
echo $! > /tmp/srpsi_real.pid
```

## 📊 监控训练

```bash
# 实时查看日志
tail -f logs/ablation_srpsi_real_jules.log

# 检查完成的 epoch 数
grep -c "Epoch [0-9]*/80" logs/ablation_srpsi_real_jules.log

# 查看最新 loss
tail -5 logs/ablation_srpsi_real_jules.log | grep loss

# 检查进程状态
ps aux | grep train.py
```

## ⏱️ 时间预估

- 环境设置：5-10 分钟
- 数据生成：5-10 分钟（如果需要）
- **训练时间：约 15 小时**

## 📦 训练完成后

模型会保存在：
```
outputs/ablation_srpsi_real/best.pth
```

下载到本地：
```bash
# 在 Mac 上执行
scp jules@JULES_IP:~/srpsi-engine-tiny/outputs/ablation_srpsi_real/best.pth \
    outputs/
```

## 🆘 遇到问题？

### 检查数据文件
```bash
ls -lh data/burgers_1d.npy
# 应该显示 ~117 MB
```

### 验证 PyTorch 安装
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 重新生成数据
```bash
rm data/burgers_1d.npy
python src/data_gen.py
```

### 停止训练
```bash
kill $(cat /tmp/srpsi_real.pid)
```

---

**准备好了吗？选择方案 A 或 B 开始吧！** 🚀
