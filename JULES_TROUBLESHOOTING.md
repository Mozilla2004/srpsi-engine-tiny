# Jules Training Troubleshooting Guide

## 🚨 Current Issue

Training stopped at Epoch 8/80 with error.

## 🔍 Diagnostic Steps

### Step 1: Check Error Log

```bash
# View last 100 lines of log
tail -100 logs/ablation_srpsi_real_jules.log

# Look for errors
grep -i "error\|exception\|traceback" logs/ablation_srpsi_real_jules.log
```

### Step 2: Identify Error Type

#### If you see: "CUDA out of memory"
→ Use Solution A (Reduce batch size)

#### If you see: "Loss is NaN" or "Loss is Inf"
→ Use Solution B (Reduce learning rate)

#### If you see: "File not found"
→ Use Solution C (Regenerate data)

#### If you see: "Killed"
→ Use Solution A (Reduce batch size)

---

## ✅ Solutions

### Solution A: Reduce Batch Size (Most Likely)

**Problem**: GPU memory insufficient

**Fix**: Create optimized config

```bash
cd ~/srpsi-engine-tiny

# The optimized config is already created
cat config/burgers_jules.yaml

# Restart with smaller batch size
nohup python src/train.py \
  --config config/burgers_jules.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real \
  > logs/ablation_srpsi_real_jules_restart.log 2>&1 &

echo $! > /tmp/srpsi_real_jules.pid
```

### Solution B: Reduce Learning Rate

**Problem**: Numerical instability

**Fix**: Create low-LR config

```bash
cat > config/burgers_low_lr.yaml << 'EOF'
inherit: default.yaml

training:
  batch_size: 16
  epochs: 80
  lr: 0.00005  # Half learning rate
  grad_clip: 0.3  # Stricter clipping
EOF

nohup python src/train.py \
  --config config/burgers_low_lr.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real \
  > logs/ablation_srpsi_real_jules_lowlr.log 2>&1 &
```

### Solution C: Regenerate Data

**Problem**: Data corruption

**Fix**:
```bash
rm data/burgers_1d.npy
python src/data_gen.py
```

---

## 🚀 Recommended Restart Procedure

### Option 1: Resume from Checkpoint (If Available)

```bash
cd ~/srpsi-engine-tiny

# Check if checkpoint exists
ls -lh outputs/ablation_srpsi_real/srpsi_real/checkpoints/

# If checkpoint exists, modify training to resume
# (Requires code modification to support resume)

# Otherwise, restart from scratch with optimized config
nohup python src/train.py \
  --config config/burgers_jules.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real_restart \
  --epochs 80 \
  > logs/ablation_srpsi_real_jules_optimized.log 2>&1 &

echo $! > /tmp/srpsi_real_jules_optimized.pid

# Monitor
sleep 5
tail -f logs/ablation_srpsi_real_jules_optimized.log
```

### Option 2: Start Fresh (Recommended)

```bash
cd ~/srpsi-engine-tiny

# Stop any existing processes
kill $(cat /tmp/srpsi_real.pid) 2>/dev/null

# Start with optimized config
nohup python src/train.py \
  --config config/burgers_jules.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real_v2 \
  --epochs 80 \
  > logs/ablation_srpsi_real_jules_v2.log 2>&1 &

PID=$!
echo $PID > /tmp/srpsi_real_v2.pid

echo "✓ Restarted with PID: $PID"
echo "✓ Batch size: 16 (vs 32)"
echo "✓ Output: outputs/ablation_srpsi_real_v2"

# Wait and check
sleep 10
tail -50 logs/ablation_srpsi_real_jules_v2.log
```

---

## 📊 Memory Optimization Details

### Before (Failed)
```
Batch Size: 32
Memory Usage: ~8-10 GB (estimated)
Status: ❌ Out of Memory
```

### After (Optimized)
```
Batch Size: 16
Memory Usage: ~4-5 GB (estimated)
Status: ✅ Should work
Training Time: +20-30% (slightly longer but stable)
```

---

## 🔔 Monitoring Commands

### Real-time Monitoring
```bash
# Watch training progress
tail -f logs/ablation_srpsi_real_jules_v2.log

# Check epoch count
grep -c "Epoch [0-9]*/80" logs/ablation_srpsi_real_jules_v2.log

# Check latest loss
tail -5 logs/ablation_srpsi_real_jules_v2.log | grep loss

# Monitor GPU memory
watch -n 5 nvidia-smi
```

---

## 💡 Alternative: Reduce Epochs

If memory is still an issue, consider training for fewer epochs:

```bash
nohup python src/train.py \
  --config config/burgers_jules.yaml \
  --model srpsi_real \
  --output outputs/ablation_srpsi_real_40ep \
  --epochs 40 \  # Only 40 epochs
  > logs/ablation_srpsi_real_jules_40ep.log 2>&1 &
```

**Trade-off**:
- 40 epochs vs 80 epochs
- Time saved: ~7 hours
- Performance: Slightly higher final loss
- Still useful for ablation study comparison

---

## 📞 Next Steps

1. **Run diagnostics**: `tail -100 logs/ablation_srpsi_real_jules.log`
2. **Identify error**: Which error message?
3. **Apply solution**: Choose Solution A, B, or C
4. **Restart training**: Use recommended commands above
5. **Monitor closely**: Check logs every few minutes

---

## ⏱️ Time Estimates

| Scenario | Time to Complete |
|----------|-------------------|
| Resume from epoch 8 | ~10 hours (72 epochs) |
| Restart with optimized config | ~15 hours (80 epochs) |
| Restart with 40 epochs | ~7 hours |

**Recommendation**: Restart with optimized config (15 hours is acceptable)
