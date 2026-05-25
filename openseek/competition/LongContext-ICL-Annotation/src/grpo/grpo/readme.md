1、先使用base版本获取容易预测错误的数据

```shell
# 任务5
python async-openseek-5-base.py
# 任务6
python async-openseek-6-base.py
# 任务7
python async-openseek-7-base.py
```



2、将错误数据制作为GRPO的训练集

```shell
# 错误数据名称
async-openseek-5-base_errors.jsonl
async-openseek-6-base_errors.jsonl
async-openseek-7-base_errors.jsonl

python handle.py
```

训练集格式如下

```json
[
  {
    "prompt": "Category: GRAMMAR \nClue: Every complete sentence must have a subject & this part containing the verb",
    "answer": "a predicate"
  },...
]
```

3、开始训练

```
python train.py
```

4、输出的经验结果

```
# experiences.json
```

5、再次预测

将json内容放进base版本的提示词中

```shell
# 任务5
python async-openseek-5-grpo.py
# 任务6
python async-openseek-7-grpo.py
# 任务7
python async-openseek-7-grpo.py
```



