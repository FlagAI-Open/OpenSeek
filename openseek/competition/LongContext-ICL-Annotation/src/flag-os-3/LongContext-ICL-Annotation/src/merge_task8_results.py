#!/usr/bin/env python
"""
关联任务8原始test_samples和已有标注结果，缺失结果填""，保持原始顺序
输出到./outputs/openseek-8-v2.jsonl
"""
import json
import os
from config import TASK_FILES, TASK8_OUTPUT_V1, TASK8_OUTPUT_V2

def main():
    # 从统一配置读取路径
    RAW_DATA_PATH = TASK_FILES[8]
    EXISTING_RESULT_PATH = TASK8_OUTPUT_V1
    OUTPUT_PATH = TASK8_OUTPUT_V2

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. 读取原始任务8数据，获取所有test_samples（保持原始顺序）
    print(f"📥 读取原始任务8数据: {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    test_samples = raw_data.get('test_samples', [])
    total_samples = len(test_samples)
    print(f"✅ 原始test_samples总数: {total_samples}个")

    # 2. 读取已有标注结果，建立ID到prediction的映射
    existing_preds = {}
    if os.path.exists(EXISTING_RESULT_PATH):
        print(f"📥 读取已有标注结果: {EXISTING_RESULT_PATH}")
        with open(EXISTING_RESULT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    sample_id = record.get('test_sample_id')
                    prediction = record.get('prediction', '')
                    if sample_id is not None:
                        existing_preds[sample_id] = prediction
                except Exception as e:
                    print(f"⚠️  忽略无效行: {str(e)}")
        print(f"✅ 已读取有效标注结果: {len(existing_preds)}个")
    else:
        print(f"⚠️  未找到已有标注结果文件 {EXISTING_RESULT_PATH}，所有结果将填为空字符串")

    # 3. 按原始顺序生成关联结果
    print(f"🔄 正在关联结果...")
    output_records = []
    filled_count = 0
    empty_count = 0

    for sample in test_samples:
        sample_id = sample.get('id')
        # 优先用已有标注，没有的填空字符串
        prediction = existing_preds.get(sample_id, "")
        output_records.append({
            "test_sample_id": sample_id,
            "prediction": prediction
        })
        if prediction:
            filled_count += 1
        else:
            empty_count += 1

    # 4. 保存结果到v2.jsonl
    print(f"💾 正在保存结果到: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 5. 打印统计信息
    print(f"\n{'='*60}")
    print(f"📊 关联完成统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  已有标注结果: {filled_count} 个 ({filled_count/total_samples*100:.1f}%)")
    print(f"  空标注结果: {empty_count} 个 ({empty_count/total_samples*100:.1f}%)")
    print(f"✅ 结果已保存到: {OUTPUT_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
