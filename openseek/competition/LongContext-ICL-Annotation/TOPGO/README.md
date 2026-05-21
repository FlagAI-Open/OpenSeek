# FlagOS OpenSeek 赛道三 - TOPGO团队

## 快速开始

### 1. 克隆仓库
```bash
cd /home
git clone https://gitee.com/anbeime/topgo-openseek.git
cd topgo-openseek
```

### 2. 安装依赖
```bash
pip install openai tqdm requests
# 可选: pip install transformers
```

### 3. 获取数据（从OpenSeek仓库）
```bash
# 如果GitHub无法访问，设置代理
export HF_ENDPOINT=https://hf-mirror.com

git clone https://github.com/FlagAI-Open/OpenSeek.git OpenSeek_temp
mkdir -p data
cp -r OpenSeek_temp/openseek/competition/LongContext-ICL-Annotation/data/* data/
rm -rf OpenSeek_temp
```

### 4. 配置API
```bash
# 华为Ascend环境
export QWEN_API_BASE="http://localhost:9010/v1/"
export QWEN_API_KEY="EMPTY"
```

### 5. 运行任务
```bash
cd src
python main.py --task_id 1 --max_input_length 10000 --log_path_prefix ../outputs/
```

## 项目结构

```
topgo-openseek/
├── data/           # 数据目录（需从OpenSeek获取）
├── src/            # 源代码
│   ├── main.py     # 主程序
│   └── method.py   # 标注方法
├── outputs/        # 输出目录
├── run_all.sh      # 一键运行脚本
├── requirements.txt # 依赖列表
└── README.md       # 本文件
```

## 团队信息

- 团队名称: TOPGO智能
- 比赛赛道: FlagOS开放计算全球挑战赛 - 赛道三：自动数据标注
