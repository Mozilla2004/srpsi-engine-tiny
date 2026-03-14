# 上传到 GitHub + Jules 训练指南

## 📦 第一步：在本地初始化 Git 仓库

```bash
# 在 srpsi-engine-tiny 目录下执行
cd /Users/luxiangrong/ClaudeCode/my-project/GenCLI+Claude/projects/srpsi-engine-tiny

# 初始化新的 git 仓库（独立于父仓库）
git init

# 添加所有必要文件
git add .gitignore
git add README.md
git add requirements.txt
git add src/
git add config/
git add data/
git add scripts/
git add *.md

# 检查将要提交的文件
git status

# 提交
git commit -m "Initial commit: SRΨ-Engine Tiny Ablation Study"
```

## 🌐 第二步：在 GitHub 创建仓库

1. 访问 https://github.com/new
2. 仓库名称：`srpsi-engine-tiny`
3. 设置为 **Private**（避免公开实验数据）
4. **不要**勾选 "Add a README file"（我们已有 README.md）
5. 点击 "Create repository"

## 📤 第三步：推送到 GitHub

```bash
# 添加远程仓库（替换成你的用户名）
git remote add origin https://github.com/YOUR_USERNAME/srpsi-engine-tiny.git

# 推送到 main 分支
git branch -M main
git push -u origin main
```

## 🤖 第四步：Jules 上拉取并训练

### 在 Jules 上执行：

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/srpsi-engine-tiny.git
cd srpsi-engine-tiny

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__} ready')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. 开始训练 Exp2 (SRΨ Real-only)
python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real \
  > logs/ablation_srpsi_real_jules.log 2>&1 &

# 6. （可选）同时训练 Exp3 (SRΨ w/o R)
python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_no_r \
  --output outputs/ablation_srpsi_no_r \
  > logs/ablation_srpsi_no_r_jules.log 2>&1 &
```

## 📊 第五步：监控训练进度

```bash
# 实时查看日志
tail -f logs/ablation_srpsi_real_jules.log
tail -f logs/ablation_srpsi_no_r_jules.log

# 检查 checkpoint 是否生成
ls -lh outputs/ablation_srpsi_*/checkpoints/
```

## 🔄 第六步：训练完成后同步结果

```bash
# 在 Jules 上找到 best checkpoint
ls -lh outputs/ablation_srpsi_real_best.pth
ls -lh outputs/ablation_srpsi_no_r_best.pth

# 下载到本地（方法 1：scp）
scp jules@jules_ip:~/srpsi-engine-tiny/outputs/ablation_srpsi_real_best.pth \
       /Users/luxiangrong/.../srpsi-engine-tiny/outputs/

# 方法 2：GitHub Release（推荐）
# - 在 GitHub 上创建 Release
# - 上传 checkpoint 文件
# - 在本地下载
```

## 📋 文件清单

### ✅ 已包含的文件（会上传到 GitHub）

```
srpsi-engine-tiny/
├── .gitignore              # 排除不必要的文件
├── README.md               # 项目说明
├── requirements.txt        # Python 依赖
├── src/                    # 源代码
│   ├── models/            # 所有模型定义
│   ├── datasets.py        # 数据加载
│   ├── metrics.py         # 评估指标
│   ├── plot.py            # 可视化
│   └── train.py           # 训练脚本
├── config/                 # 配置文件
│   └── burgers.yaml       # 1D Burgers 方程配置
├── data/                   # 训练数据
│   └── burgers_1d.npy    # 117MB 数据文件
├── scripts/               # 工具脚本
│   └── analyze_ablation_results.py
└── *.md                   # 文档和报告
```

### ❌ 排除的文件（不会上传，.gitignore）

```
venv/                  # 虚拟环境（太大，可重建）
.venv_srpsi/          # 虚拟环境
outputs/              # 训练输出（checkpoint 等训练完再下载）
logs/                 # 训练日志
*.log                 # 临时日志
results/              # 评估结果
```

## ⚠️ 注意事项

### 1. 大文件处理

`data/burgers_1d.npy` 是 117MB，GitHub 默认限制是 100MB。

**解决方案 A（推荐）**：使用 Git LFS
```bash
# 安装 Git LFS
git lfs install

# 追踪 .npy 文件
git lfs track "*.npy"

# 重新添加并提交
git add .gitattributes
git add data/burgers_1d.npy
git commit -m "Add data file with LFS"
git push
```

**解决方案 B（简单）**：手动上传
1. 跳过 `data/` 目录，不上传到 GitHub
2. 在 Jules 上手动下载 `burgers_1d.npy`：
   ```bash
   wget https://your-dropbox-link/burgers_1d.npy
   # 或从其他机器 scp
   ```

### 2. 训练时间估计

- 单个实验：~15 小时（CPU）
- 两个实验并行（Jules）：~15 小时
- 建议：**只跑 Exp2**，让 Mac 继续跑 Exp3

### 3. 资源分配

| 平台 | 实验 | 原因 |
|------|------|------|
| **Jules** | Exp2 (SRΨ Real) | 最简单，最稳定，适合远程 |
| **Mac** | Exp3 (SRΨ w/o R) | 你可以监控 |
| **Mac** | Exp4, Exp5 | 继续训练中 |

## 🎯 推荐方案

### 方案 A：保守策略
```bash
# 只在 Jules 上跑 Exp2
# Mac 继续跑 Exp3, Exp4, Exp5
# 总时间：15 小时
```

### 方案 B：激进策略（推荐）
```bash
# Jules 跑 Exp2
# Mac 停止 Exp3，让 Jules 也跑 Exp3
# Windows 专注跑其他任务
# 总时间：15 小时，但结果更可靠
```

## ✨ 快速命令总结

```bash
# === 本地（Mac） ===
cd /Users/luxiangrong/.../srpsi-engine-tiny
git init
git add .gitignore README.md requirements.txt src/ config/ data/ scripts/ *.md
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/srpsi-engine-tiny.git
git push -u origin main

# === Jules ===
git clone https://github.com/YOUR_USERNAME/srpsi-engine-tiny.git
cd srpsi-engine-tiny
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py --config config/burgers.yaml --model srpsi_real \
  > logs/ablation_srpsi_real_jules.log 2>&1 &
```

---

**准备好了吗？开始吧！** 🚀
