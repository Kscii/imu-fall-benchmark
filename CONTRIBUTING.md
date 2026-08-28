# 协作说明

## 当前范围

本仓库当前只维护 WSL2/CUDA 上的时序滑动窗口跌倒 benchmark。新增任务前，先在 Issue 中写清楚目的、数据范围、合同变化、预期产物和验收方法。位置分类、Android、标注 UI、云端服务和产品告警策略应分别在对应仓库或后续 workstream 中处理。

## 提交前检查

在 WSL2 中执行：

```bash
./benchmark test
./benchmark validate-data
./benchmark plan configs/experiments/temporal_smoke_v1.yaml
```

只修改纯 Python/合同单元测试时，也可以在其他 Linux 开发机执行：

```bash
uv sync --dev
uv run ruff check src tests
uv run pytest -q
```

正式训练、doctor 和 smoke 仍必须在 WSL2 + NVIDIA CUDA 上执行。

## 数据与合同规则

- 不提交 `*.h5`、cache、run、模型 checkpoint、云凭据或本地 `TODO.md`；
- 数据通过 GCS 的不可变 snapshot + `current.json` 管理；
- 不允许原地替换已经发布的 snapshot 对象；
- base 数据使用 fold `0..4`，team 数据只能使用训练专用 fold `-1`；
- 修改采样率、窗口、步长、标签、fold 或指标时，必须提升相应 schema/contract/config 版本并增加回归测试；
- 不允许为了得到更好结果而静默改变 seed、split、阈值选择或数据过滤规则；
- 结果必须保留 resolved config、来源、环境、日志和 machine-readable 指标。

## 代码规则

- 保持公开 CLI 简单，优先扩展现有 `setup / data / doctor / test / smoke / plan / run / report`；
- 配置写入 `configs/`，不可把协议常量散落在命令行参数中；
- 注释和代码标识使用英文，组员文档当前使用中文；
- 自动生成内容写入 `IMU_BENCH_WORK_ROOT`，不写入仓库；
- 提交应保持单一目的，避免把数据合同、模型逻辑和文档重写混在一个无法审查的 commit 中。

## 结果表述

Smoke 只能证明流程可运行。单 fold pilot 是工程验证，不是最终模型结论。正式比较至少需要团队先确认数据范围、fold、seed、指标、阈值协议和需要保留的 artefact。`fall_score` 在完成单独校准前不能称为概率。
