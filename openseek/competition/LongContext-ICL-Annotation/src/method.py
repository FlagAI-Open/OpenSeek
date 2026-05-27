import re

from collections import Counter
from transformers import AutoTokenizer

from api_client import ChatClient
from const import CONST_LIMIT_OUTPUT_MAX_TOKENS


# 利用ICL提取Operation的Prompt
def make_icl_prompt(task_description: str, icl_sample_data: str):
    temperature = 0.7
    prompt = f"你是一名专业的数据标注专家，请仔细阅读理解示例数据的任务目标(在==左侧的<input>与</input>之间)与任务结果(在==右侧的<output>与</output>之间)，首先利用示例数据按照任务描述的要求对任务目标与任务结果进行推理和推导，并且必须使用使用逆向思维、跳跃思维和横向关联以避免深陷在反复思考或递归循环之中，然后分析提取出从任务目标得到任务结果所必须执行的在10个以内的关键操作，并且必须是在对所有示例数据需要的普适通用的关键操作，并且必须与任务描述强相关的，并且必须与示例数据弱相关的，然后表示为言简意赅的纯英文提示词短语，不能包含引号、逗号、分号、句号、括号、横线、表情等标点符号或特殊符号，不能包含数字或项目符号，不能是无关的或冗余的，不能是示例数据或解释内容，不能是假设或推测，最后必须用<operation>与</operation>包裹每个关键操作提示词。\
        任务描述: {task_description} \
        示例数据: {icl_sample_data}"

    return prompt, temperature


MORES = {
    'text': '',
    'array': '',
    'number': '',
    'phrase': '',
    'code': '必须使用编程语言类库方法生成完整的符合人们使用经验的文件格式的可直接执行代码脚本，'
}

# 利用Operation和Test获取Result的Prompt
def make_test_prompt(task_description: str, task_input: str, core_operations: str, return_data_type:str):
    assert return_data_type in MORES.keys(), f"return_data_type must be one of {MORES.keys()}, but your is error: {return_data_type}"

    more = MORES[return_data_type]
    temperature = 0.3
    prompt = f"你是一名永远都保持冷静与理性的解题高手，请仔细阅读理解任务描述、任务目标和关键操作(在每个<operation>与</operation>之间)，你必须要一直保持冷静与理性地解决问题，不要让外界信息影响到你的情绪和判断，首先参考任务描述与关键操作进行联想和思考，从中找出最相关的关键操作, 并且根据你的最佳经验对当前任务目标补充缺失的关键操作，然后按照任务描述的要求对任务目标进行推理和推导，并且必须使用逆向思维、跳跃思维和横向关联以避免深陷在反复思考或无限循环之中，并且按照任务描述的规范生成任务目标的任务结果，并且其格式与数据类型必须严格符合任务描述的规范与要求，结果不能是假设或推测，结果不能是重复罗嗦或粗浅描述，结果不能包含的多余的空格空行，{more} 必须将任务结果全用英文表达作为标准答案，最后必须填写在<result>与</result>之间。\
        任务描述: {task_description} \
        任务目标: {task_input} \
        关键操作: {core_operations}"

    return prompt, temperature


# 同步单个提交ICL推理请求到模型服务端，提交的数据是一个批次的ICL示例
def refer_get_operation_for_icl(task_description: str, icl_sample_data: str, max_tokens=10240):
    prompt, temperature = make_icl_prompt(task_description, icl_sample_data)
    max_tokens = min(max_tokens, CONST_LIMIT_OUTPUT_MAX_TOKENS)
    answer, state = ChatClient.request_llm_api_stream(prompt, max_tokens, temperature)
    if state == False:
        return answer, state

    return parse_operations(answer)


# 同步单个提交测试推理请求到模型服务端，提交的数据是一个批次的Operation，一个测试数据
def refer_get_result_for_test(task_description: str, task_input: str, core_operations: str, return_data_type: str, max_tokens=10240):
    prompt, temperature = make_test_prompt(task_description, task_input, core_operations, return_data_type)
    max_tokens = min(max_tokens, CONST_LIMIT_OUTPUT_MAX_TOKENS)
    answer, state = ChatClient.request_llm_api_stream(prompt, max_tokens, temperature)
    if state == False:
        return answer, state

    return parse_result(answer, return_data_type)


# 异步并发提交ICL推理请求到模型服务端，提交的数据是多个批次的ICL示例
def batch_refer_get_operation_for_icl(task_description: str, icl_sample_data_list: list, max_tokens=10240):
    temperature = None
    prompts = []
    for icl_sample_data in icl_sample_data_list:
        prompt, temperature = make_icl_prompt(task_description, icl_sample_data)
        prompts.append(prompt)

    # 请求模型服务端
    max_tokens = min(max_tokens, CONST_LIMIT_OUTPUT_MAX_TOKENS)
    answers = ChatClient.batch_request_llm_api_stream(prompts, max_tokens, temperature)

    outputs = []
    for answer, state in answers:
        if state == False:
            outputs.append((answer, state))
        else:
            outputs.append(parse_operations(answer))

    return outputs


# 异步并发提交测试推理请求到模型服务端，提交的数据是一个批次的Oepration，多个测试数据
def batch_refer_get_result_for_test(task_description: str, task_input_list: list, core_operations: str, return_data_type: str, max_tokens=10240):
    temperature = None
    prompts = []
    for task_input in task_input_list:
        prompt, temperature = make_test_prompt(task_description, task_input, core_operations, return_data_type)
        prompts.append(prompt)

    max_tokens = min(max_tokens, CONST_LIMIT_OUTPUT_MAX_TOKENS)
    answers = ChatClient.batch_request_llm_api_stream(prompts, max_tokens, temperature)

    outputs = []
    for answer, state in answers:
        if state == False:
            outputs.append((answer, state))
        else:
            outputs.append(parse_result(answer, return_data_type))

    return outputs


# 批量拼接ICL示例数据（implementation of Long-Context Data Annotation）
def select_examples(icl_examples: list[dict], tokenizer: AutoTokenizer, start: int, stop: int, target_length=8192) -> str:
    """
        Select examples from icl_examples to fit into the target context length (适配Qwen3-4B的token计算).
        icl_examples:
            A list of examples, where each example is a dict with keys 'input' and 'output' (no 'length' needed).
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review"}``,
        tokenizer:
            Qwen3-4B的分词器
        start:
            最小起始位置,范围[0,len(icl_examples)-1]
        stop:
            最大结束位置,范围[0,len(icl_examples)-1]
        target_length:
            上下文长度限制（Qwen3-4B的上下文窗口默认是8k/32k，根据实际调整，若比较严格则建议为8192）
    """
    total = len(icl_examples)
    assert total > 0 and start >=0 and stop >= start and start < total and stop < total, f"must be `total > 0 and start >=0 and stop >= start and start < total and stop < total`, error: {start}, {stop}, {total}"

    examples_str, token_num = "", 0

    # 遍历所有示例，基于Qwen3-4B的tokenizer计算token数
    while start <= stop:
        example = icl_examples[start]

        # 提取input和output(兼容output是列表的情况)
        input_text = example['input']
        output_text = example['output'][0]

        # 核心: 用Qwen3-4B的tokenizer计算input+output的token数(替代原length键)
        # encode返回token id列表，len即为token数
        input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
        length = input_tokens + output_tokens  # 等效原示例的length值

        # 单条数据长度超过长度限制，跳过该条数据
        if length > target_length:
            start += 1
            continue

        # 校验当前示例是否能加入(总长度不超限制)
        if length + token_num <= target_length:
            # 累加总token数: 示例文本长度 + 格式符号的token数(<input>7 + </input>==<output>18 + </output>9)
            token_num += (length + 7 + 18 + 9)
            # 拼接示例字符串
            examples_str += f"<input>{input_text}</input>==<output>{output_text}</output>"
            start += 1
        else:
            # 累计数据长度超过长度限制，当前截止拼接
            break

    # 返回已拼接的示例和已选数量，作为当前批次
    return examples_str, start


# 批量拼接关键操作提示词operations
def select_operations(task_operations: list[dict], tokenizer: AutoTokenizer, start: int, stop: int, target_length=8192) -> str:
    """
        Select operations from task_operations to fit into the target context length (适配Qwen3-4B的token计算).
        task_operations:
            A list of operations, where each operation is a string.
        tokenizer:
            Qwen3-4B的分词器
        start:
            最小起始位置,范围[0,len(task_operations)-1]
        stop:
            最大结束位置,范围[0,len(task_operations)-1]
        target_length:
            上下文长度限制（Qwen3-4B的上下文窗口默认是8k/32k，根据实际调整，若比较严格则建议为8192）
    """
    total = len(task_operations)
    assert total > 0 and start >=0 and stop >= start and start < total and stop < total, f"must be `total > 0 and start >=0 and stop >= start and start < total and stop < total`, error: {start}, {stop}, {total}"

    operations_str, token_num = "", 0
    # 遍历所有示例，基于Qwen3-4B的tokenizer计算token数
    while start <= stop:
        operation = task_operations[start]

        # 核心: 用Qwen3-4B的tokenizer计算input+output的token数(替代原length键)
        # encode返回token id列表，len即为token数
        length = len(tokenizer.encode(operation, add_special_tokens=False))

        # 单条数据长度超过长度限制，跳过该条数据
        if length > target_length:
            start += 1
            continue

        # 校验当前示例是否能加入(总长度不超限制)
        if length + token_num <= target_length:
            # 累加总token数: 示例文本长度 + 格式符号的token数(<operation>11 + </operation>12)
            token_num += (length + 11 + 12)
            # 拼接单个示例字符串
            operations_str += f"<operation>{operation}</operation>"
            start += 1
        else:
            # 累计数据长度超过长度限制，当前截止拼接
            break

    # 返回已拼接的示例和已选数量，作为当前批次
    return operations_str, start


# 获取关键操作的提示词
def parse_operations(answer: str) -> list:
    """
    提取字符串中<operation>标签内的所有内容(字符串形式)，统计出现次数最多的内容
    :answer: 包含<operation>标签的原始字符串
    :return: 去掉标签或分隔符之后的文本数组
    """
    if answer == None:
        return None, False

    operations = __parse_operations_from_str(answer)
    if len(operations) == 0:
        return None, False

    content_counter = Counter(operations)
    if not content_counter:
        return None, False

    # 提取最高频次的前10个
    max_count = max(content_counter.values())
    sel_count = max_count - 10
    operations = [ content.strip() for content, count in content_counter.items() if count > sel_count ]

    return operations, True


# 获取测试数据的预测结果
def parse_result(answer: str, return_data_type: str):
    assert return_data_type in MORES.keys(), f"return_data_type must be one of {MORES.keys()}, but your is error: {return_data_type}"

    if answer == None:
        return None, False

    answer = answer.strip()
    if answer[0:8] == '<result>':
        answer = answer[8:None].strip()

    if answer[-9:None] == '</result>':
        answer = answer[0:-9].strip()

    match return_data_type:
        # 数字
        case 'number':
            arr = __parse_number(answer)

        # 英文短语
        case 'phrase':
            arr = __parse_phrase(answer)

        # 包含多种符号的任意英文数组
        case 'array':
            arr = __parse_array(answer)

        # 包含多种编程的任意英文字符串
        case 'code':
            langs = ['python', 'java', 'go', 'sql', 'php', 'rust', 'zig', 'swift', 'javascript', 'c', 'cpp', 'csharp', 'ruby', 'sh', 'bash']
            pattern =  r'(' + '|'.join([f'```{lang}\n?' for lang in langs]) + ')'
            arr = re.findall(pattern, answer, re.I)
            if len(arr) > 0:
                answer = (answer.split(arr[-1])[-1]).strip()
                if answer[-3:None] == '```':
                    answer = answer[None:-3]
            return answer, True

        # 包含多种符号的任意英文字符串
        case 'text':
            return answer, True

        case _:
            return answer, True

    data_counter = Counter(arr)
    if not data_counter:
        return None, False

    # 提取最高频次的数值作为最终答案
    max_count = max(data_counter.values())
    result = [ data.strip() for data, count in data_counter.items() if count == max_count ]

    return result[0], True


# 提取关键操作的英文短语
def __parse_operations_from_str(text: str):
    if text.find('<operation>') > -1:
        arr = []
        # 首个是空字符或者无关信息
        for s in text.split('<operation>')[1 : None]:
            arr.append(s.split('</operation>')[0].strip())
    else:
        # 标点符号通常用作分隔符（'"-_\s/ 往往也是操作短语的组成部分）
        arr = re.split(r'[.,;:!?@#$%&*()\[\]{}<>~`^\\\n\t]', text.split('</operation>')[0].strip())

    return [ s.strip() for s in arr if len(s) > 3 and not re.search(r'\d|(inappropriate)', s) ]


# 提取英文数值,例如: -0.9, 12.33, 10_000
def __parse_number(text: str) -> list:
    return re.findall(r'-?\d+(?:[\d._]*\d)?', text)


# 提取英文短语
def __parse_phrase(text: str) -> list:
    return re.findall(r'\b(?:[a-zA-Z0-9]+\.?\s?)+(?:-[A-Za-z0-9]+\.?\s?)*\b', text)


# 提取英文数组
def __parse_array(text: str) -> list:
    return re.findall(r"\[[a-zA-Z0-9.,;:!?@#$%&*()\[\]{}<>|/~`'^\"\\\-\s\n\t]+\]", text.strip())



if __name__=="__main__":
    r = parse_operations('sas-1,sasa_1\nsasa_1.234;sdsd"ds')
    print(r)

    r = parse_operations('<operation>sas-1,sasa_1\nsasa_1.234;sdsd"ds</operation>')
    print(r)

    r = __parse_number('sas-1,sasa_1\nsasa_1.234;sdsd"ds')
    print(r)

    r = __parse_phrase('sas-1,sasa_1\nsasa_1.234;sdsd"ds')
    print(r)

    r = __parse_array('sa["s-1","sasa_1\nsasa_1.234","[12,3]"')
    print(r)

    a = '[274, 190, 74, 4, 3]'
    r = __parse_array(a)
    print(r)

    r = parse_result(a, 'array')
    print(r)


    import os
    from transformers import AutoTokenizer

    qwen_tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.expanduser('~'), '.cache/modelscope/hub/models/Qwen/Qwen3-4B'))

    r,pos = select_examples([{
            'input': "A",
            'output': ["a"],
        }, {
            'input': "B",
            'output': ["b"],
        }, ], qwen_tokenizer)
    print(r,pos)

    r,pos = select_operations(['a', 'b', 'c'], qwen_tokenizer)
    print(r,pos)
