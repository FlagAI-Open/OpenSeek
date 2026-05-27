# 数据目录约定

建议直接把官方 `LongContext-ICL-Annotation/data` 目录导入到这里：

```text
data/raw/
└── openseek/
    ├── openseek-1_closest_integers.json
    ├── openseek-2_count_nouns_verbs.json
    ├── ...
    └── openseek-8_kernel_generation.json
```

导入命令：

```bash
./scripts/import_official_data.sh /path/to/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data
```

每个官方 JSON 文件内含：

- `task_id`
- `task_name`
- `Definition`
- `examples`
- `test_samples`
- `License`

如果后续官方文件名变化，调整 `configs/datasets/registry.yaml` 即可。

`data/processed/` 可用于缓存：

- 召回后的示例索引
- 分块后的长上下文输入
- 多轮摘要记忆
