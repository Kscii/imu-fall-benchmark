# 标注平台与 Benchmark 数据契约

状态：`dataset_handoff pre-v1`；`experiment_catalog` 与 `model_release` 已冻结 v1
规范源：`Kscii/imu-fall-benchmark`  
读者：数据采集/标注平台与 benchmark 的开发者

本文只定义两个仓库之间交换的持久化制品，不规定网页 API、页面布局、数据库结构或部署方式。字段名、枚举值、对象键和文件名使用英文；规范说明使用中文。

## 1. 模块与版本

三个模块独立演进：

| 模块 | 当前版本 | 作用 |
| --- | --- | --- |
| `dataset_handoff` | `0.1.0` | 标注平台向 benchmark 交接训练 HDF5 |
| `experiment_catalog` | `1.0.0` | benchmark 向标注平台发布实验及逐模型 ONNX 证据 |
| `model_release` | `1.0.0` | benchmark 向标注平台发布固定候选模型及其验证范围 |

版本使用 SemVer。正式 `1.0.0` 之前可以删除、重命名或改变字段含义，不保证向前兼容，但两个仓库必须在同一对 PR 中同步规范、实现、测试和锁文件。

读取 v1 制品时，读取方接受相同 major 的兼容 minor/patch：只要求本版本的必需字段是制品字段的子集，并忽略未知可选字段。某个模块第一次正式发布后，该模块冻结为 `1.0.0`：

- patch：澄清文字或修复不改变语义的实现；
- minor：只添加可选字段，读取方必须忽略未知可选字段；
- major：破坏性修改，必须明确批准，并让旧 major 的已发布制品继续可读；
- 已发布对象不可原地修改；更正必须使用新 ID 和新对象键。

第一批正式团队数据冻结 `dataset_handoff`；第一份正式实验目录和第一份正式模型分别冻结另外两个模块。它们不要求同时进入 v1。

## 2. 通用规则

- 所有 JSON 使用 UTF-8；时间使用 UTC ISO 8601；SHA-256 使用 64 位小写十六进制。
- 文件描述符统一包含 `filename`、`object_key`、`size_bytes`、`sha256` 和 `content_type`。
- 发布方先上传内容文件，验证大小与 SHA-256，最后创建不可变的 `metadata.json` 或 `manifest.json` 标记。
- 稳定 ID 仅允许 `[A-Za-z0-9._-]`；对象键不得含绝对路径或 `..`。
- `state.json` 是标注平台拥有的可变展示状态，只允许 `available` 与 `deprecated`，不属于 benchmark 交接证据，也不得改变不可变 metadata。
- 页面只负责浏览、解释和下载，不在浏览器中执行 ONNX 推理，也不自动声明 `current`、`recommended` 或 `best`。

## 3. `dataset_handoff` 0.1.0

### 3.1 制品与对象布局

一次团队快照包含：

```text
benchmark-datasets/team/cw12eu/<snapshot_id>/datasets/cw12eu.h5
benchmark-datasets/team/cw12eu/<snapshot_id>/manifest.json
benchmark-datasets/team/cw12eu/current.json
```

`cw12eu.h5` 使用内部物理格式 `imu_schema_version = 3.1.0`。物理 HDF5 schema 与仓库间 handoff 版本是两件事；升级其中之一不自动升级另一个。

`manifest.json` 至少包含：

```json
{
  "schema_version": "imu_benchmark_dataset_manifest_v1",
  "handoff_contract_version": "0.1.0",
  "kind": "team",
  "snapshot_id": "...",
  "files": [
    {
      "filename": "cw12eu.h5",
      "object_key": "benchmark-datasets/team/cw12eu/.../datasets/cw12eu.h5",
      "size_bytes": 1,
      "sha256": "...",
      "content_type": "application/x-hdf5"
    }
  ]
}
```

`current.json` 必须同时记录 `snapshot_id`、`manifest_object`、`manifest_sha256` 和 `handoff_contract_version`。流程应先创建并验证不可变快照，再切换 `current.json`；不得让 current 指向尚未通过 `imu-bench data pull --team-snapshot <snapshot_id>` 与 `validate-team` 的制品。

`snapshot_id` 的内容 fingerprint 必须把 handoff 合同版本纳入身份。当前版本以已按
`participant_id`、`recording_id` 排序的 `recordings` 清单构造以下对象：

```json
{
  "handoff_contract_version": "0.1.0",
  "recordings": []
}
```

随后使用 UTF-8、JSON 排序键和紧凑分隔符（`,`、`:`）编码，计算 SHA-256，并以摘要前
24 个十六进制字符组成 `snapshot-<digest>`。同一合同和同一录制内容必须得到同一 ID；即使
录制内容不变，合同版本变化也必须得到新 ID。旧合同快照只能保留为历史证据，不得为补字段
而原地覆盖不可变 manifest。

### 3.2 HDF5 3.1.0 约束

- `/samples`：`float32 [N, 6]`，顺序为 `acceleration_x_mps2`、`acceleration_y_mps2`、`acceleration_z_mps2`、`angular_velocity_x_radps`、`angular_velocity_y_radps`、`angular_velocity_z_radps`，采样率 25 Hz，均为 SI 单位。
- `/sequences`：包含 `sample_start`、`sample_stop`、`source_file`、`participant_id`、`recording_id`、`body_location`、`activity_code`、`is_fall`、`supervision_kind`、`source_sampling_rate_hz`。
- `/annotations`：包含 `sequence_index`、`kind`、`start_sample`、`stop_sample`、`code`。
- `activity` 与 `exclude` 使用半开区间 `[start_sample, stop_sample)`；`onset` 与 `impact` 是点事件，要求 `start_sample == stop_sample`。
- temporal 跌倒序列中的 `onset` 与 `impact` 必须落在对应跌倒 activity 区间内；每个跌倒区间最多各有一个 onset 和 impact。

原始 BLE 包、原始计数、视频、同步 review、标签管理和校准证据由标注平台保留，不属于此 handoff。benchmark 只能把交接的 SI HDF5 当作训练输入，不能从 HDF5 反推或改写原始证据。

## 4. `experiment_catalog` 1.0.0

### 4.1 对象布局

```text
benchmark-model-catalog/experiments/<publication_id>/onnx/<artifact_id>.onnx
benchmark-model-catalog/experiments/<publication_id>/result.tar.gz   # 可选
benchmark-model-catalog/experiments/<publication_id>/metadata.json
benchmark-model-catalog/experiments/<publication_id>/state.json      # 平台拥有
```

现有 `benchmark-results/...` result v1 继续作为不可变的完整计算证据。experiment catalog 是独立索引：可以引用既有 result 的 ID、manifest SHA-256 和 bundle SHA-256，但不得覆盖或升级旧 result manifest。

首次上线页面已经发布的 `imu_experiment_catalog_v0 / 0.1.0` 属于冻结前历史对象。读取方必须继续通过管理 API 以只读方式识别并明确标记为 `legacy_pre_v1`；缺少 `metric_split` 或 `selection_eligible` 时只能保守解释为 `metric_split = "test"`、`selection_eligible = false`，不得补写原 metadata。迁移到 v1 必须使用新的 `publication_id` 和对象前缀重新发布；本次正式迁移 ID 固定为 `formal_baseline_temporal_core_onnx_v1-bfef2ab3d903-catalog-v1`。v1 完成远程验证后，平台把旧发布的 `state.json` 改为 `deprecated` 并从普通模型页面隐藏，但管理 API 仍须允许审计；不得删除或覆盖旧制品。

`metadata.json` 的根字段：

- `schema_version = "imu_experiment_catalog_v1"`
- `contract_version` 的 major 必须为 `1`
- `publication_id`、`run_id`、`experiment_id`、`evidence_level`、`created_at_utc`
- `source`、`data`、`evaluation_fingerprint`、`scheduled_jobs`
- `methods`：方法级均值与样本标准差，只用于说明实验结果
- `artifacts`：每个 fold/seed 模型的输入输出、指标、判定规则、运行时证据和 ONNX 文件描述符
- `result_evidence`：既有 result manifest/bundle 的引用；完整 TAR 可通过 `result_bundle` 作为可选直接下载
- `known_limitations`

所有从 CV test fold 汇总的指标必须显式记录 `metric_split = "test"` 与 `selection_eligible = false`。平台可以展示这些评估结果，但不得把它们用于模型、阈值或触发策略选择。正式候选模型的选择证据必须来自独立的 validation OOF 计算，并证明每个开发参与者只出现一次；该证明的必要摘要必须进入模型发布的 `metadata.json`，不能只存在于未发布的本地文件。

每个 artifact 必须完整记录：

- 身份：`artifact_id`、`model_id`、`training_recipe`、`fold`、`seed`、`backend`；
- `input` 与 `output`，包括形状、dtype、通道、单位、采样率和预处理位置；
- `metrics.window`、`metrics.event` 及各 alarm policy 的指标；
- `decision.score_threshold`：数值、选择方法、选择 split、比较符 `>=`；
- `decision.trigger_policies`：所有评估过的策略，含 `required_positive_windows`、`lookback_windows`、`consecutive`、`cooldown_seconds`、`reference_policy`、`validation_pareto`；
- `decision.anchor = "window_end"`；
- `parity` 与 ONNX/runtime 版本证据；
- `onnx` 文件描述符。

平台必须把 `formal_cv` 与 `engineering` 明确区分，不能把工程预检描述成正式模型比较。实验页面可以下载逐 artifact ONNX、整体 metadata 和可选 result TAR，但不得据 test 指标自动推荐一个 fold，也不得默认按 test 指标排序、突出或标记优胜方法。用户可以主动选择指标排序，但排序控件附近必须持续显示 `metric_split = "test"` 与 `selection_eligible = false` 的含义。

## 5. `model_release` 1.0.0

### 5.1 两文件合同

模型发布载荷恰好包含两个不可变文件：

```text
benchmark-model-catalog/models/<release_id>/model.onnx
benchmark-model-catalog/models/<release_id>/metadata.json
```

标注平台可以在同一前缀额外维护 `state.json`，但它是平台拥有的生命周期 sidecar，不属于模型发布载荷，也不得被下载方当成模型身份或训练证据。

`metadata.json` 是最后写入的发布标记，根字段：

- `schema_version = "imu_model_release_v1"`
- `contract_version` 的 major 必须为 `1`
- `release_id`、`model_code`、`name`、`created_at_utc`
- `release_stage = "research_candidate"`；不得暗示 best、recommended 或产品认证
- `source.selection_evidence` 与 `source.final_training`：分别记录 validation-only 选择和全量 final-refit 来源
- `data`：训练/验证数据快照与 fingerprint
- `input`、`output`、`preprocessing`
- `windowing`：2 秒窗口、发布特定的正数 `inference_interval_seconds`、`anchor = "window_end"` 和序列/gap reset 语义；推理节奏必须与选择证据中的 source stride 一致
- `decision`：固定的 `score_threshold`、一个固定 `trigger_policy`、`status = "provisional_validation_derived"`
- `metrics`、`validation`、`known_limitations`
- `verification.golden_fixtures`：至少包括静止、ADL-like 与 impact-like 三个不含真实参与者数据的合成输入，以及期望 `fall_score` 和容差
- `model`：`model.onnx` 的完整文件描述符

`source.selection_evidence` 必须直接嵌入以下可审计摘要：

- 来源实验的 `source_run_id`、源码 commit、`model_id`、`training_recipe`、数据/切分 fingerprint；
- `selection_scope = "validation_only_oof"`、`metric_split = "validation_oof"` 与 `selection_eligible = true`；
- 正数 `source_stride_seconds`；它是生成 OOF 选择证据时实际使用的决策步长，必须与模型
  `windowing.inference_interval_seconds` 完全一致；训练窗口步长则从来源数据合同写入
  `windowing.training_stride_seconds`，两者不得互相替代；
- participant proof，包括 `status = "PASS"`、参与者总数、`appearances_per_participant = 1`、各 validation fold 的参与者数和 assignment SHA-256；
- 阈值选择方法、触发策略选择方法及其确定性 tie-break；对应数值指标放在 `metrics`；
- 若引用额外 selection artifact，必须给出可验证的不可变对象键、大小和 SHA-256。只写本地路径或文件名无效，且外部 artifact 不得取代上述内嵌摘要。

`source.final_training` 必须记录干净源码 commit、随机种子、固定 epoch 的来源、训练范围和实际 epoch；final-refit 只能使用已经完成 validation-only 选择后确定的配置，不得再读取 test 指标调整模型。`metrics` 必须明确 `metric_split = "validation_oof"`、`selection_eligible = true` 和 `final_model_independently_evaluated = false`，避免把选择证据误解为 final-refit 模型的独立准确率。

最终 ONNX 输入为未经 runtime 标准化的 SI `float32 [batch, 50, 6]`，顺序固定为 `ax, ay, az, gx, gy, gz`，25 Hz、`sensor_local`、保留重力；训练集均值和尺度必须嵌入 ONNX graph。输出为未校准的 `fall_score`，不能称为概率。

发布前必须通过 ONNX checker 与全量 final-training window 的 Python ONNX Runtime parity，并验证 metadata 中的 model SHA-256、大小、dtype、shape、输入输出名和实际 ONNX 一致。验证器还必须实际运行 metadata 内嵌的合成 golden fixtures，而不能只信任 `PASS` 字符串。`external_runtime` 与 `device_replay` 可以明确为 `not_tested`；Python parity 不代表 Android、Core ML、Windows 或嵌入式设备已经可部署。

模型 release 表示已经固定了一套“分数阈值 + 触发策略”，但不等于产品安全认证，也不保证真实跌倒检测性能。平台必须显式展示 `release_stage`、指标 split、是否独立评估、`external_runtime`、`device_replay` 和 `known_limitations`；`not_tested` 必须与 `PASS` 使用不同文案和视觉状态。页面不得使用“最终模型”“最佳模型”或“推荐模型”指代 `research_candidate`。

## 6. 跨仓库同步

标注平台保存本文的只读生成副本，并保存锁文件：

```json
{
  "upstream_repository": "Kscii/imu-fall-benchmark",
  "upstream_commit": "<40-hex>",
  "canonical_path": "docs/contracts/annotation-benchmark-contract.zh-CN.md",
  "sha256": "<64-hex>",
  "module_versions": {
    "dataset_handoff": "0.1.0",
    "experiment_catalog": "1.0.0",
    "model_release": "1.0.0"
  }
}
```

更新步骤：先在 benchmark PR 修改规范与生产方；取得该 commit 后，在标注平台 PR 同步完全相同的正文、更新 lock、消费者和测试。任何一侧发现字段不足，都可以发起规范 PR；但 canonical 历史始终在 benchmark 仓库，标注平台副本不得单独编辑。
