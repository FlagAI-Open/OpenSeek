## Submission Format (JSONL + ZIP)

Below is the standard format for submitting model predictions. Please save your predictions into **eight `.jsonl` files** (one per task), then **package them into a single `.zip` archive** and upload it to the FlagOS platform for automatic evaluation.

---

### 1) JSONL File Content

Each `.jsonl` file consists of multiple JSON objects (**one prediction per line**).  
Each prediction must contain the following two fields:

- `test_sample_id`: corresponds to the sample `id` in the competition dataset.
- `prediction`: the model’s predicted result for that sample.

**Single-line example:**
```json
{"test_sample_id":"openseek-1-ed5ac69191204cd4bfb0ca41bc7f197f","prediction":"..."}
```

### 2) ZIP Archive Requirements (Mandatory)
Each submission must upload **one** `.zip` file, and the archive must contain **8** prediction files:

- Each filename must start with `openseek-[id]` (e.g., `openseek-1*.jsonl`)
- All **8** tasks correspond to **8** `.jsonl` files for automated scoring.

> Recommendation: Make sure the `.zip` archive contains these **8** `.jsonl` files directly (no nested folders), and avoid including any unrelated extra files to prevent evaluation parsing issues.

---

## 提交格式说明（JSONL + ZIP）

以下为标准的模型预测结果提交规范。请将模型预测结果分别保存为 **8 个 `.jsonl` 文件**，并将它们 **打包为一个 `.zip` 压缩包** 后上传至 FlagOS 平台进行自动评测。

---

### 1) JSONL 文件内容格式

每个 `.jsonl` 文件由多行 JSON 对象组成（**一行一个预测结果**）。  
每条预测必须包含以下两个字段：

- `test_sample_id`：对应赛题数据中的样本 `id`
- `prediction`：模型对该样本的预测结果

**单行示例：**
```json
{"test_sample_id":"openseek-1-ed5ac69191204cd4bfb0ca41bc7f197f","prediction":"..."}
```

### 2) ZIP 压缩包要求（必须满足）
每次提交需上传 **一个** `.zip` 文件，且该压缩包必须同时包含 **8** 个预测文件：

- 文件名需以 `openseek-[id]` 开头（例如 `openseek-1*.jsonl`）
- 共 **8** 个任务各对应 **8** 个 `.jsonl` 文件，用于自动化评分

> 建议：确保压缩包内直接包含这 **8** 个 `.jsonl` 文件（不嵌套文件夹），并避免额外无关文件，以免影响评测解析。