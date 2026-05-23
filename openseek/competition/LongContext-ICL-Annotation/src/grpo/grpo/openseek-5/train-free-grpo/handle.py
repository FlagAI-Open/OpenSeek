import json

def load_indented_json(file_path: str):
    """
    加载【多行缩进】的连续JSON对象文件（适配你的数据格式）
    :param file_path: JSON文件路径
    :return: 列表，每个元素是一个JSON字典
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取整个文件内容
        content = f.read()
        decoder = json.JSONDecoder()
        current_index = 0
        total_length = len(content)

        # 循环解析所有JSON对象
        while current_index < total_length:
            # 跳过空格、换行、缩进等空白字符
            while current_index < total_length and content[current_index].isspace():
                current_index += 1
            if current_index >= total_length:
                break

            # 解析一个完整的JSON对象
            json_obj, current_index = decoder.raw_decode(content, current_index)
            data_list.append(json_obj)

    return data_list

# ====================== 使用示例 ======================
if __name__ == '__main__':
    # 替换为你的文件路径（如 test.jsonl / data.json）
    result = []
    file = r"./async-openseek-5-base.jsonl"
    # 获取预测正确的数据
    result_true = load_indented_json(file)
    result_true = [{"prompt": tmp["input"], "answer": tmp["ground_truth"]} for tmp in result_true if tmp["ground_truth"] == tmp["prediction"]]


    file = r"./async-openseek-5-base_errors.jsonl"
    # 加载数据
    result_false = load_indented_json(file)
    result_false = [{"prompt": tmp["input"], "answer": tmp["ground_truth"]} for tmp in result_false]

    import random
    random.seed(42)
    result += result_true[:50] + result_false[:100]
    random.shuffle(result)

    # 查看结果
    print(f"成功加载 {len(result)} 条数据")
    # 打印第一条数据（格式化输出，方便查看）
    print("\n第一条数据：")
    print(json.dumps(result[0], indent=2, ensure_ascii=False))

    with open("train_data.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))



