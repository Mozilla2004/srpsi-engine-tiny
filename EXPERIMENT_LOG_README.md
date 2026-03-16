# 实验日志系统使用指南

**版本**: v0.1
**创建日期**: 2026-03-16
**灵感来源**: spirit-coding-lab 的 Federation Protocol v0.1

---

## 📚 文件说明

### 1. EXPERIMENT_LOG.md
**作用**: 项目演化历史的叙事记录

**内容**:
- 8 个历史条目 (Entry 0001 - Entry 0008)
- 关键转折点 (T1, T2, T3)
- 实验阶段进度

**何时更新**:
- 完成一个实验后
- 产生重要发现时
- 架构发生重大变化时

**如何更新**:
```markdown
## Entry 0009

- **timestamp**: 2026-03-16T12:00:00+08:00
- **type**: experiment_result
- **phase**: physical_validation
- **summary**: 🎯 简短描述
- **details**: |
  详细描述
  - 发现 1
  - 发现 2

**交付物**: filename.md
**状态**: ✅ 完成
```

---

### 2. EXPERIMENT_STATE.json
**作用**: 当前实验状态的压缩快照

**内容**:
- 实验阶段 (experiment_phase)
- 模型性能 (models)
- 数据集状态 (datasets)
- 关键发现 (key_findings)
- TRAE 洞察 (trae_insights)
- 当前任务 (current_tasks)
- 下一步行动 (next_actions)

**何时更新**:
- 每次实验运行后
- 模型性能更新时
- 任务状态变化时

**如何更新**:
1. 更新对应字段的值
2. 更新 `timestamp`
3. 如果有新发现,添加到 `key_findings`
4. 如果任务完成,从 `current_tasks` 移到完成状态,或从 `next_actions` 移到 `current_tasks`

---

### 3. EXPERIMENT_PHASES.md
**作用**: 实验阶段的明确定义和进度跟踪

**内容**:
- Phase 1-5 的定义
- 每个阶段的目标、任务、交付物
- 关键转折点
- Milestone 完成度

**何时更新**:
- 任务完成时 (更新 checkbox `[ ]` → `[x]`)
- 阶段完成时 (更新状态、进度条)
- 新增任务时

**如何更新**:
```markdown
- [x] 已完成任务
- [ ] 待完成任务

Milestone 3: [██████████░░░░░░░░░░] 60% 🔄
```

---

## 🚀 快速开始

### 第一次使用

1. **阅读历史**:
   ```bash
   cat EXPERIMENT_LOG.md
   ```
   了解项目从开始到现在的完整历史

2. **查看当前状态**:
   ```bash
   cat EXPERIMENT_STATE.json | jq .
   ```
   了解项目现在在哪里

3. **了解阶段规划**:
   ```bash
   cat EXPERIMENT_PHASES.md
   ```
   了解项目要到哪里去

### 日常使用

#### 1. 开始新实验前
```bash
# 查看当前状态和下一步行动
cat EXPERIMENT_STATE.json | jq .next_actions
```

#### 2. 实验运行中
```bash
# 查看当前阶段和任务
cat EXPERIMENT_PHASES.md | grep -A 10 "Phase 4"
```

#### 3. 实验完成后
**步骤 1**: 更新 EXPERIMENT_LOG.md
```markdown
## Entry 0009
- **timestamp**: [当前时间]
- **type**: experiment_result
...
```

**步骤 2**: 更新 EXPERIMENT_STATE.json
```json
"models": {
  "srpsi_v2_hybrid": {
    "status": "completed",
    "performance": {...}
  }
}
```

**步骤 3**: 更新 EXPERIMENT_PHASES.md
```markdown
- [x] Physical Tests v2.0
```

---

## 📊 与 spirit-coding-lab 的对应关系

| spirit-coding-lab | srpsi-engine-tiny | 说明 |
|-------------------|-------------------|------|
| user_rules.md | - | 不适用 (科学研究项目,非 AI 智能体) |
| PHASE_RULES.md | EXPERIMENT_PHASES.md | 项目阶段定义 |
| FIELD_LAWS.md | - | 不适用 (物理定律已在代码中实现) |
| FIELD_LOG.md | EXPERIMENT_LOG.md | 演化历史记录 |
| FIELD_SNAPSHOT.json | EXPERIMENT_STATE.json | 当前状态快照 |

**关键区别**:
- spirit-coding-lab: **AI 进化观察实验室** (多智能体协作)
- srpsi-engine-tiny: **机器学习实验项目** (科学研究)

---

## 🎯 最佳实践

### 1. 及时更新
- ✅ 实验完成后立即更新
- ❌ 不要等到最后才补记录

### 2. 保持简洁
- ✅ 记录关键发现和转折点
- ❌ 不要记录所有细节

### 3. 使用 Markdown
- ✅ 用标题、列表、代码块组织内容
- ❌ 不要写大段无结构的文字

### 4. 量化结果
- ✅ 用数字说话 (MSE ↓ 6.5%)
- ❌ 不要用模糊描述 ("有所改善")

### 5. 链接文件
- ✅ 在日志中引用相关文件
- ```markdown
  **交付物**: `test_v2_physical.py`
  ```

---

## 🔧 自动化脚本 (可选)

### update_log.sh
```bash
#!/bin/bash
# 自动更新实验日志的辅助脚本

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S+08:00")
ENTRY_NUM=$(ls EXPERIMENT_LOG.md | grep -c "Entry")

echo "新的实验日志条目:"
echo "Entry Number: $((ENTRY_NUM + 1))"
echo "Timestamp: $TIMESTAMP"
echo "Type: [experiment_result|insight|milestone|paradigm_shift]"
echo "Phase: [setup|training|analysis|validation|synthesis]"
```

### check_state.sh
```bash
#!/bin/bash
# 检查当前实验状态

echo "=== 当前实验状态 ==="
cat EXPERIMENT_STATE.json | jq '.experiment_phase, .milestones'

echo ""
echo "=== 下一步行动 ==="
cat EXPERIMENT_STATE.json | jq '.next_actions'
```

---

## 📝 模板

### 新实验日志条目模板
```markdown
## Entry XXXX

- **timestamp**: YYYY-MM-DDTHH:MM:SS+08:00
- **type**: [experiment_result|insight|milestone|paradigm_shift]
- **phase**: [setup|training|analysis|validation|synthesis]
- **summary**: 🎯 简短描述
- **details**: |
  详细描述
  - 发现 1
  - 发现 2

**交付物**: filename.md
**状态**: [✅ 完成|🔄 进行中|⏳ 待开始]
```

### 新任务模板 (EXPERIMENT_PHASES.md)
```markdown
- [ ] 任务名称
  - [ ] 子任务 1
  - [ ] 子任务 2
  **目标**: ...
  **交付物**: ...
```

---

## 💡 常见问题

### Q1: 什么时候更新这三个文件?
**A**:
- **EXPERIMENT_LOG.md**: 每次完成重要实验或发现后
- **EXPERIMENT_STATE.json**: 每次实验后更新性能数据
- **EXPERIMENT_PHASES.md**: 任务完成时更新 checkbox

### Q2: 如果实验失败了,要记录吗?
**A**: 要! 负面结果也是重要的科学发现
```markdown
## Entry XXXX
- **type**: negative_result
- **summary**: ❌ 尝试 X 失败
- **details**: |
  - 原因: ...
  - 启示: ...
```

### Q3: 如何在团队协作中使用?
**A**:
1. 每个人添加自己的 Entry 时,使用自己的编号
2. 在 commit message 中引用 Entry 号
3. 定期同步和合并日志

### Q4: 需要记录所有的代码改动吗?
**A**: 不需要。只记录:
- 架构重大变化
- 新算法/新特征
- 关键 bug 修复
- 性能突破

---

**维护者**: SRΨ-Engine Tiny 实验组
**最后更新**: 2026-03-16
**版本**: v0.1
