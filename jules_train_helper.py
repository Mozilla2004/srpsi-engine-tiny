#!/usr/bin/env python3
"""
Jules 训练助手脚本
=================

自动化设置和启动训练流程

Usage:
    python jules_train_helper.py

Author: SRΨ-Engine Tiny Experiment
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """执行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"命令: {cmd}")
    print(f"{'-'*60}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ {description} - 成功")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ {description} - 失败")
        print(f"错误: {result.stderr}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     SRΨ-Engine Tiny - Jules 训练启动助手                ║
║                                                          ║
║     任务: 训练 Exp2 (SRΨ Real-only)                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    # 检查是否在项目根目录
    if not Path("src/train.py").exists():
        print("❌ 错误: 请在项目根目录 (srpsi-engine-tiny) 运行此脚本")
        sys.exit(1)

    # 第一步: 检查虚拟环境
    print("\n📦 第一步: 检查虚拟环境")
    venv_exists = Path("venv").exists()
    if venv_exists:
        print("✅ 虚拟环境已存在")
    else:
        print("⚠️  虚拟环境不存在，将创建...")
        if not run_command("python -m venv venv", "创建虚拟环境"):
            sys.exit(1)

    # 激活虚拟环境的命令（根据操作系统）
    activate_cmd = "source venv/bin/activate"  # Linux/Mac

    # 第二步: 检查依赖
    print("\n📥 第二步: 检查依赖安装")
    try:
        import torch
        print(f"✅ PyTorch 已安装: {torch.__version__}")
    except ImportError:
        print("⚠️  PyTorch 未安装，正在安装依赖...")
        if not run_command(f"{activate_cmd} && pip install --upgrade pip", "升级 pip"):
            sys.exit(1)
        if not run_command(f"{activate_cmd} && pip install -r requirements.txt", "安装依赖"):
            sys.exit(1)

    # 第三步: 检查数据文件
    print("\n📊 第三步: 检查训练数据")
    data_file = Path("data/burgers_1d.npy")

    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"✅ 数据文件已存在: {size_mb:.1f} MB")
    else:
        print("⚠️  数据文件不存在，正在生成...")
        print("   (这可能需要几分钟，请耐心等待)")

        # 创建 data 目录
        Path("data").mkdir(exist_ok=True)

        # 生成数据
        if not run_command("python src/data_gen.py", "生成训练数据"):
            print("❌ 数据生成失败")
            sys.exit(1)

        # 验证生成的文件
        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"✅ 数据生成成功: {size_mb:.1f} MB")
        else:
            print("❌ 数据文件未生成")
            sys.exit(1)

    # 第四步: 创建必要的目录
    print("\n📁 第四步: 创建输出目录")
    Path("outputs/ablation_srpsi_real").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    print("✅ 目录已创建")

    # 第五步: 启动训练
    print("\n🚀 第五步: 启动训练")
    print("="*60)
    print("配置:")
    print("  - 模型: SRΨ Real-only (Exp2)")
    print("  - 配置文件: config/burgers.yaml")
    print("  - 输出目录: outputs/ablation_srpsi_real")
    print("  - 日志文件: logs/ablation_srpsi_real_jules.log")
    print("  - 预计时间: ~15 小时")
    print("="*60)

    train_cmd = (
        f"nohup python src/train.py "
        f"--config config/burgers.yaml "
        f"--model srpsi_real "
        f"--output outputs/ablation_srpsi_real "
        f"> logs/ablation_srpsi_real_jules.log 2>&1 &"
    )

    print(f"\n执行命令:\n{train_cmd}\n")

    # 启动训练
    result = subprocess.run(
        train_cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    # 保存进程 ID
    pid_result = subprocess.run(
        "echo $!",
        shell=True,
        capture_output=True,
        text=True
    )
    pid = pid_result.stdout.strip()

    if pid:
        with open("/tmp/srpsi_real.pid", "w") as f:
            f.write(pid)
        print(f"✅ 训练已启动！进程 ID: {pid}")

    # 等待几秒后查看日志
    print("\n⏳ 等待 5 秒后检查训练状态...")
    import time
    time.sleep(5)

    # 显示日志开头
    if Path("logs/ablation_srpsi_real_jules.log").exists():
        print("\n📋 训练日志（前 50 行）:")
        print("="*60)
        log_result = subprocess.run(
            "head -50 logs/ablation_srpsi_real_jules.log",
            shell=True,
            capture_output=True,
            text=True
        )
        print(log_result.stdout)

    # 监控命令
    print("\n" + "="*60)
    print("✨ 训练已成功启动！")
    print("="*60)
    print("\n📊 监控命令:")
    print("  实时查看日志:")
    print("    tail -f logs/ablation_srpsi_real_jules.log")
    print("\n  检查进度:")
    print("    grep -c 'Epoch [0-9]*/80' logs/ablation_srpsi_real_jules.log")
    print("\n  查看最新 loss:")
    print("    tail -5 logs/ablation_srpsi_real_jules.log | grep loss")
    print("\n  检查进程状态:")
    print(f"    ps -p {pid}")
    print("\n⏹️  停止训练（如需要）:")
    print(f"    kill {pid}")
    print("\n📦 训练完成后，模型保存在:")
    print("    outputs/ablation_srpsi_real/best.pth")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
