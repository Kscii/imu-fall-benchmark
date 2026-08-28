# IMU Fall Benchmark

这是一个面向组员内部使用的六轴 IMU 跌倒检测 benchmark。仓库负责版本控制训练代码、数据合同、fold、实验配置、测试和结果格式；HDF5 数据不进入 Git，而是从项目的 Google Cloud Storage（GCS）bucket 下载。

当前版本只处理一个任务：使用带时间区间标注的数据训练和评估因果滑动窗口跌倒分类器。位置分类、recording-level MIL、Android 实时告警策略和自动部署不在本仓库当前范围内。

模型输出的 `fall_score` 是分类分数，除非后续单独完成概率校准，否则不能把它描述为“跌倒概率”。

## 运行环境

正式训练和评估只支持：

- Windows 11 + WSL2；
- NVIDIA CUDA GPU，参考机器为 RTX 4070 SUPER 12 GB；
- 仓库位于 WSL 的 Linux 文件系统，例如 `~/projects/imu-fall-benchmark`，不能放在 `/mnt/c`；
- Python 3.12、CUDA 12.9、RAPIDS 26.08、XGBoost 3.4.0 和 PyTorch 2.8.0。

Windows 只需要安装支持 WSL2 的 NVIDIA 驱动，不要在 WSL 内再次安装 Linux 显卡驱动。

## 第一次使用

先在 WSL2 安装最小系统依赖：

```bash
sudo apt update
sudo apt install --yes git curl ca-certificates tar
```

然后把仓库 clone 到 WSL 文件系统并执行：

```bash
cd ~/projects/imu-fall-benchmark
./benchmark setup
./benchmark data pull
./benchmark doctor
./benchmark smoke
```

`setup` 会在 `~/imu-fall-work` 中安装或复用固定版本的 Miniforge、Google Cloud CLI 和 CUDA Python 环境，不会修改系统 Python。第一次安装通常需要 20–60 分钟，并建议预留至少 25 GiB 空间。

`data pull` 首次执行时会要求登录 Google 账号。它只下载 `current.json` 指向的不可变 snapshot，随后逐文件核对 SHA-256、HDF5 v3.1 结构、逻辑指纹和统计值。仓库不需要 Git LFS。每个 WSL 用户只需登录一次：首次必须在可交互的 WSL 终端直接执行该命令；无 TTY 的 SSH/Codex 自动化不会复制宿主机凭据，也不会在无法输入验证码时继续。登录完成后，后续拉取可以自动运行。

## 日常命令

```bash
./benchmark data status
./benchmark data pull
./benchmark validate-data
./benchmark test
./benchmark doctor
./benchmark smoke
```

- `data status`：比较本地 active manifest 和远程 current pointer；
- `data pull`：原子下载和激活最新 base/team snapshot；
- `validate-data`：检查全部 HDF5、hash、统计值与 participant fold；
- `test`：执行 Ruff 与 pytest；
- `doctor`：检查 WSL2、CUDA、GPU 和七种模型后端；
- `smoke`：运行经过限量的 fold-0 七模型端到端测试，可复用兼容 checkpoint。

## 数据合同

机器可读合同位于 [`configs/contracts/imu_benchmark_contract_v2.json`](configs/contracts/imu_benchmark_contract_v2.json)，关键规则如下：

- HDF5 schema：`3.1.0`；
- 输入：25 Hz、六轴 IMU、`float32`、sensor-local 坐标系、保留重力；
- 窗口：50 帧，即 2 秒；
- 步长：物理时间 0.5 秒；在 25 Hz 上使用 half-up 网格，起点为 `0, 13, 25, 38, ...`；
- 决策时间：窗口最后一帧；
- 正样本：决策时间落在半开跌倒区间 `[fall_start, fall_stop)` 内；
- 与明确 `exclude` 区间相交的窗口会被排除；
- 已经越过跌倒区间终点、但仍包含跌倒尾部的窗口会被排除，避免用跌倒后的信息制造负样本；
- 当前事件检测规则：一个正窗口即视为检测到该跌倒。N-of-M、冷却时间和告警合并属于后续产品策略实验。

原始 public snapshot 包含 9 个 HDF5 shard。KFall 和 UNIVRFall 提供可用的跌倒时间区间；只有 recording 标注的数据中，ADL recording 可以提供负窗口，但 fall recording 因缺少区间不会被当作时序正样本。后续标注平台产生的 CW12EU 数据必须包含时序区间。

## Fold 与 team 数据

公共数据使用固定的 participant-disjoint 五折：测试 fold 为 `k`，验证 fold 为 `(k + 1) mod 5`，其余三折训练。阈值只根据验证 fold 的 Balanced Accuracy 选择。

标注平台持续产生的 team snapshot 使用 `fold_id = -1`：

- 可以加入每个 fold 的训练集；
- 永远不能进入验证集或测试集；
- 这样有限的内部设备数据可帮助训练，但不会让模型在自己见过的 participant/recording 上得到虚高的评估结果。

任何数据 snapshot、fold、合同或实验配置变化都会改变 cache/run 指纹。旧 run 不会被静默复用。

## 实验配置

当前保留三种入口：

| 配置 | 用途 |
|---|---|
| `temporal_smoke_v1.yaml` | 限量端到端检查，不作为模型结论 |
| `kfall_fold0_regression_v1.yaml` | KFall 单折完整回归参考 |
| `all_temporal_fold0_pilot_v1.yaml` | 全部可用时序数据的单折工程 pilot |

先查看计划，再运行：

```bash
./benchmark plan configs/experiments/all_temporal_fold0_pilot_v1.yaml
./benchmark run configs/experiments/all_temporal_fold0_pilot_v1.yaml --resume
```

七种模型为 Threshold Impact、cuML Logistic Regression、cuML Random Forest、CUDA XGBoost、PyTorch 1D CNN、PyTorch LSTM 和 PyTorch CNN-LSTM。三个表格模型读取 158 个工程特征；三个深度时序模型读取 `50 × 6` 原始窗口；Threshold Impact 直接读取原始窗口。

所有配置都固定 fold、seed、精度、epoch/patience 与数据合同。单 fold pilot 是工程证据，不等于最终多折模型验证。

## 输出和恢复

默认生成内容全部保存在仓库外：

```text
~/imu-fall-work/
├── data/                 # 已验证的 immutable snapshots 与 active.json
├── cache/                # 50×6 窗口和 158 特征的内容寻址缓存
├── envs/                 # 固定依赖环境
├── runs/<run-id>/        # 配置、checkpoint、指标、日志和报告
└── toolchains/           # Miniforge 与 Google Cloud CLI
```

可以把 `IMU_BENCH_WORK_ROOT` 设置为其他绝对路径。每个 job 完成后会原子写入 checkpoint；中断后使用 `--resume`，只会复用配置和数据指纹完全一致的结果。

主要输出包括 `resolved_config.yaml`、`environment.json`、`provenance.json`、`events.jsonl`、`jobs/*.npz`、`metrics.csv`、`event_metrics.csv`、`subgroup_metrics.csv`、`performance.json` 和 `report.md`。

## 数据发布结构

默认 bucket 为 `gs://soft3888-label`，也可通过 `IMU_BENCH_DATA_BUCKET=gs://bucket-name` 覆盖。benchmark 只使用以下前缀：

```text
benchmark-datasets/
├── base/<snapshot-id>/datasets/*.h5
├── base/<snapshot-id>/manifest.json
├── base/current.json
├── team/cw12eu/<snapshot-id>/datasets/*.h5
├── team/cw12eu/<snapshot-id>/manifest.json
└── team/cw12eu/current.json
```

`benchmark-datasets/` 已启用为 GCS managed folder，组员的只读 IAM 只应绑定到该资源，不能授予
整个 bucket 的 Viewer。snapshot 对象不可覆盖；`current.json` 只是一个小型显式指针。发布公共
base 是维护操作：

```bash
./benchmark data publish-base --source-dir /path/to/reviewed/imu_25hz
```

日常用户只执行 `data pull`，不需要 bucket 写权限。任何凭据、登录缓存、HDF5、运行结果和本地 `TODO.md` 都不能提交到 Git。

协作规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
