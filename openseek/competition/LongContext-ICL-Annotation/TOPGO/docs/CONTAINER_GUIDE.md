#!/bin/bash
# ====================================================================
# FlagOS OpenSeek 赛道三 - 容器内一键运行指南
# ====================================================================

echo "============================================================"
echo "FlagOS OpenSeek 赛道三 - 容器内操作指南"
echo "============================================================"

# 步骤1：检查并启动vLLM服务
echo ""
echo "[步骤1] 检查vLLM服务..."
echo ""

if curl -s http://localhost:8000/v1/models | grep -q "Qwen"; then
    echo "✅ vLLM服务正常运行"
    vllm_model=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data'][0]['id'])")
    echo "   当前模型: $vllm_model"

    if [[ "$vllm_model" != *"/home/Qwen/Qwen3-4B"* ]]; then
        echo ""
        echo "⚠️  警告：vLLM未使用正确的Qwen3-4B模型！"
        echo "   需要重启vLLM服务..."
        pkill -f vllm
        sleep 5

        nohup python -m vllm.entrypoints.openai.api_server \
          --model /home/Qwen/Qwen3-4B \
          --host 0.0.0.0 --port 8000 \
          --gpu-memory-utilization 0.9 \
          --trust-remote-code \
          --max-model-len 32768 \
          > /tmp/vllm.log 2>&1 &

        echo "   等待30秒..."
        sleep 30
    fi
else
    echo "❌ vLLM服务未运行，正在启动..."

    nohup python -m vllm.entrypoints.openai.api_server \
      --model /home/Qwen/Qwen3-4B \
      --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      > /tmp/vllm.log 2>&1 &

    echo "   等待30秒..."
    sleep 30
fi

# 步骤2：拉取最新代码
echo ""
echo "[步骤2] 拉取最新代码..."
echo ""

cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 步骤3：验证修复
echo ""
echo "[步骤3] 验证代码修复..."
echo ""

if grep -q "Qwen2.5-3B-Instruct" src/main.py; then
    echo "❌ 警告：代码仍然包含Qwen2.5-3B-Instruct！"
    echo "   正在手动修复..."
    sed -i "/Qwen2.5-3B-Instruct/d" src/main.py
    echo "✅ 已修复"
else
    echo "✅ 代码已正确（不包含违规模型）"
fi

# 步骤4：设置环境变量
echo ""
echo "[步骤4] 设置环境变量..."
echo ""

export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

echo "✅ 环境变量已设置"

# 步骤5：清理旧结果（可选）
echo ""
echo "[步骤5] 检查输出目录..."
echo ""

mkdir -p outputs

# 检查现有结果
task_files=""
for i in 1 2 3 4 5 6 7 8; do
    file="outputs/openseek-${i}-v1.jsonl"
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  任务$i: 已存在 ($lines 行)"
    else
        echo "  任务$i: 未生成"
        task_files="$task_files $i"
    fi
done

# 步骤6：运行任务
echo ""
echo "[步骤6] 运行任务..."
echo ""

if [ -z "$task_files" ]; then
    echo "✅ 所有任务已完成，无需重新运行"
    echo ""
    echo "如需重新运行特定任务，执行："
    echo "  python main.py --task_id <任务ID> --max_input_length 15000 --log_path_prefix ../outputs/"
else
    echo "需要运行的任务: $task_files"
    echo ""
    read -p "是否运行缺失的任务？(y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd src

        for task_id in $task_files; do
            echo ""
            echo "========== 开始任务 $task_id =========="
            START=$(date +%s)

            python main.py --task_id $task_id --max_input_length 15000 --log_path_prefix ../outputs/

            END=$(date +%s)
            DURATION=$((END - START))

            echo "任务 $task_id 完成！耗时: $((DURATION / 60)) 分钟"
        done

        cd ..
    fi
fi

# 步骤7：清理BOM并验证
echo ""
echo "[步骤7] 清理BOM并验证文件..."
echo ""

cd /home/topgo-openseek

# 下载并运行清理脚本
cat > clean_and_validate.py << 'PYEOF'
import os
import json
import glob

def remove_bom(filepath):
    """移除BOM并验证"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    valid_lines = []
    for line in lines:
        try:
            data = json.loads(line)
            valid_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
        except:
            pass

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)

    return len(valid_lines)

# 处理所有文件
for filepath in sorted(glob.glob('outputs/openseek-*-v*.jsonl')):
    if os.path.exists(filepath):
        count = remove_bom(filepath)
        print(f"✅ {os.path.basename(filepath)}: {count} 行")

print("\n所有文件已清理BOM并验证")
PYEOF

python3 clean_and_validate.py

# 步骤8：最终统计
echo ""
echo "[步骤8] 最终统计..."
echo ""

python3 << 'PYEOF'
import json
import glob

total_samples = 0
total_valid = 0

print("任务完成情况:")
print("-" * 60)

for filepath in sorted(glob.glob('outputs/openseek-*-v1.jsonl')):
    task_id = filepath.split('-')[1]

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    nulls = sum(1 for l in lines if '"prediction": null' in l)
    valid = total - nulls

    total_samples += total
    total_valid += valid

    status = "✅" if valid/total > 0.8 else ("⚠️" if valid/total > 0.5 else "❌")
    print(f"{status} 任务{task_id}: {valid:3d}/{total:3d} 有效 ({valid*100/total:5.1f}%)")

print("-" * 60)
print(f"总计: {total_valid}/{total_samples} 有效 ({total_valid*100/total_samples:.1f}%)")
PYEOF

echo ""
echo "============================================================"
echo "操作完成！"
echo "============================================================"
echo ""
echo "输出文件位置: /home/topgo-openseek/outputs/"
echo ""
echo "文件列表:"
ls -lh outputs/openseek-*-v1.jsonl
echo ""
echo "下载结果到本地（在本地执行）:"
echo "  scp root@<服务器IP>:/home/topgo-openseek/outputs/openseek-*-v1.jsonl ./outputs/"

# ====================================================================
# LDR集成 - Local Deep Research 增强功能
# ====================================================================
echo ""
echo "============================================================"
echo "LDR集成 - Local Deep Research 增强功能"
echo "============================================================"

# 步骤9：验证LDR模块
echo ""
echo "[步骤9] 验证LDR模块..."
echo ""

cd /home/topgo-openseek

# 检查LDR模块是否存在
if [ -f "src/evidence_verifier.py" ] && [ -f "src/dynamic_retriever.py" ]; then
    echo "✅ LDR模块已集成"
    echo ""
    echo "可用模块:"
    echo "  - src.evidence_verifier: 证据验证模块"
    echo "  - src.dynamic_retriever: 动态检索模块"
    echo "  - src.SelfEvolvingKnowledgeBase: 自进化知识库"
    echo ""
    
    # 测试导入
    python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/topgo-openseek')

try:
    from src import DynamicRetriever, EnhancedQualityInspector
    from src.evidence_verifier import EvidenceVerifier
    from src.dynamic_retriever import SelfEvolvingKnowledgeBase
    print("✅ 所有LDR模块导入成功")
except ImportException as e:
    print(f"❌ 导入失败: {e}")
PYEOF
else
    echo "⚠️ LDR模块未找到，请拉取最新代码"
    echo "   git fetch origin && git reset --hard origin/master"
fi

# 步骤10：LDR功能测试
echo ""
echo "[步骤10] LDR功能测试（可选）..."
echo ""

read -p "是否运行LDR功能测试？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/topgo-openseek')

print("=" * 60)
print("LDR功能测试")
print("=" * 60)

# 测试1: EvidenceVerifier
print("\n[测试1] EvidenceVerifier")
try:
    from src.evidence_verifier import EvidenceVerifier, EnhancedQualityInspector
    
    verifier = EvidenceVerifier(max_duration=10)
    enhanced_qc = EnhancedQualityInspector(evidence_verifier=verifier)
    
    # 模拟验证
    decision = enhanced_qc.verify_entity_with_evidence(
        entity="测试实体",
        entity_type="organization",
        context="这是一个测试上下文",
        task_domain="general"
    )
    
    print(f"  ✅ 决策: {decision.decision}, 置信度: {decision.confidence:.2f}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# 测试2: DynamicRetriever (需要embedding模型)
print("\n[测试2] DynamicRetriever")
try:
    from src.dynamic_retriever import DynamicRetriever, SelfEvolvingKnowledgeBase
    
    # 注意：完整测试需要embedding模型
    print("  ⚠️ 需要embedding模型才能完整测试")
    print("  模块已正确加载")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# 测试3: SelfEvolvingKnowledgeBase
print("\n[测试3] SelfEvolvingKnowledgeBase")
try:
    from src.dynamic_retriever import SelfEvolvingKnowledgeBase
    
    # 模拟知识库操作
    print("  ✅ 知识库模块已加载")
    print("  使用方法:")
    print("    kb = SelfEvolvingKnowledgeBase(embedding_model)")
    print("    kb.add_annotation_trajectory(text, annotation, quality_score=0.95)")
    print("    kb.retrieve_similar_cases(query, top_k=3)")
except Exception as e:
    print(f"  ❌ 失败: {e}")

print("\n" + "=" * 60)
print("LDR功能测试完成")
print("=" * 60)
PYEOF
fi

echo ""
echo "============================================================"
echo "LDR集成验证完成"
echo "============================================================"
echo ""
echo "技术文档位置: /home/topgo-openseek/docs/LDR_INTEGRATION.md"
echo "容器使用指南: /home/topgo-openseek/docs/LDR_CONTAINER_GUIDE.md"
