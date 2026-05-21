import os
import re
import time
import requests
import asyncio
import aiohttp

from openai import OpenAI, AsyncOpenAI, APIConnectionError, AuthenticationError, APIError, BadRequestError

from const import CONST_DEBUG_SHOW_STREAM, CONST_DEBUG_SHOW_ASNSWER


serve_api_key = os.environ.get('SERVE_API_KEY', '')
serve_base_url = os.environ.get('SERVE_BASE_URL', 'http://localhost:2026/v1')
serve_model_name = os.environ.get('SERVE_MODEL_NAME', 'Qwen/Qwen3-4B')
debug_print_stream = os.environ.get('DEBUG_PRINT_STREAM', 0)

# url格式处理
serve_base_url = serve_base_url.split('v1')[0].rstrip('/') + '/v1'


# 使用CHAT模版请求模型服务端
class ChatClient:

    # 多个并发请求模型服务（流式）
    def batch_request_llm_api_stream(prompts: list, max_tokens=10240, temperature=0.6, enable_thinking=True):
        answers = []
        texts = asyncio.run(ChatClient.__batch_request_chat_api_stream(prompts, max_tokens, temperature, enable_thinking))
        for i, (text, state) in enumerate(texts):
            if debug_print_stream == str(CONST_DEBUG_SHOW_ASNSWER) or debug_print_stream == CONST_DEBUG_SHOW_ASNSWER:
                print('text, state = ', text, state)

            # 若格式不符合要求，则多请求几次尽量提高成功率
            if state == False or text.find('<output_answer>') == -1:
                time.sleep(3)
                text, state = ChatClient.request_llm_api_stream(prompts[i], max_tokens, temperature, enable_thinking)
                if debug_print_stream == str(CONST_DEBUG_SHOW_ASNSWER) or debug_print_stream == CONST_DEBUG_SHOW_ASNSWER:
                    print('text, state = ', text, state)

            if state == False:
                text, _ = ChatClient.__parse_answer(text)
                answers.append((text, state))
            else:
                answers.append(ChatClient.__parse_answer(text))

        return answers


    # 单个请求模型服务（流式）
    def request_llm_api_stream(prompt: str, max_tokens=10240, temperature=0.7, enable_thinking=True):
        text, state = asyncio.run(ChatClient.__request_chat_api_stream(prompt, max_tokens, temperature, enable_thinking))
        if debug_print_stream == str(CONST_DEBUG_SHOW_ASNSWER) or debug_print_stream == CONST_DEBUG_SHOW_ASNSWER:
            print('text, state = ', text, state)

        # 若格式不符合要求，则多请求一次尽量提高成功率
        if state == False or text.find('<output_answer>') == -1:
            time.sleep(3)
            text, state = asyncio.run(ChatClient.__request_chat_api_stream(prompt, max_tokens, temperature, enable_thinking, 512))
            if debug_print_stream == str(CONST_DEBUG_SHOW_ASNSWER) or debug_print_stream == CONST_DEBUG_SHOW_ASNSWER:
                print('text, state = ', text, state)

        if state == False:
            text, _ = ChatClient.__parse_answer(text)
            return text, state
        else:
            return ChatClient.__parse_answer(text)


    # 单个请求模型服务
    def request_llm_api(prompt: str, max_tokens=10240, temperature=0.7, enable_thinking=True):
        text, state = ChatClient.__request_chat_api(prompt, max_tokens, temperature, enable_thinking)
        if debug_print_stream == str(FLAG_SHOW_ASNSWER) or debug_print_stream == FLAG_SHOW_ASNSWER:
            print('text, state = ', text, state)

        # 若格式不符合要求，则多请求一次尽量提高成功率
        if state == False or text.find('<output_answer>') == -1:
            time.sleep(3)
            text, state = ChatClient.__request_chat_api(prompt, max_tokens, temperature, enable_thinking, 512)
            if debug_print_stream == str(FLAG_SHOW_ASNSWER) or debug_print_stream == FLAG_SHOW_ASNSWER:
                print('text, state = ', text, state)

        if state == False:
            text, _ = ChatClient.__parse_answer(text)
            return text, state
        else:
            return ChatClient.__parse_answer(text)


    # 获取任务结果英文文本（可以包含分隔符和特殊标签）
    def __parse_answer(text: str) -> str:
        """
        提取字符串中<output_answer>标签内的所有内容(字符串形式)，统计出现次数最多的内容
        :text: 包含<output_answer>标签的原始字符串
        :return: 最后一个完整<output_answer></output_answer>之内的数据
        """
        # print("text:", text)
        if text == None:
            return None, False

        if text.find('<output_answer>') > -1:
            output_answer = text.split('<output_answer>')[-1].split('</output_answer>')[0].strip()

        else:
            # 返回格式可能不符合预期，则用正则提取英文内容，
            # 在Prompts已经要求返回英文结果(可包含常见的特殊英文字符，支持返回编程代码)
            pre_answers = re.findall(r"[a-zA-Z0-9.,;:!?@#$%&*()\[\]{}<>|/~`'^\"\\\-\s\n\t]+", text)
            if len(pre_answers) == 0:
                return None, False

            # 排除空白
            answers = []
            for a in pre_answers:
                a = a.strip()
                if a != '':
                    answers.append(a)
            if len(answers) == 0:
                return None, False

            # 但此时只要提取最后一个英文内容 (答案通常都是输出在最后)
            output_answer = answers[-1]

        # print("output_answer:", output_answer)
        return output_answer, True


    # 模型服务直接输出
    def __request_chat_api(prompt: str, max_tokens=10240, temperature=0.6, enable_thinking=True, len_for_exception_response=0):
        client = OpenAI(
            api_key = serve_api_key,
            base_url = serve_base_url.rstrip('/'),
            timeout = 60*60*24,
        )

        messages = [
            {
                "role": "system",
                "content": "你是一名无所不知无所不能的全能高手，请仔细理解任务描述和任务目标，展开所有联想找到最佳策略和方法，尽你所能生成用户期望的标准答案，返回结果的格式也须符合用户要求。"
            },
            {
                "role": "user",
                "content": str(prompt) if enable_thinking else str(prompt) + '/no_think', # 必须强制转一次字符串
            }
        ]

        try:
            response = client.chat.completions.create(
                model = serve_model_name,
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
                stream = False, # 是否流式输出方式
            )

            text = response.choices[0].message.content.strip()
            # print("Response-full_text:", text)

            # 处理异常数据：长度截断、者胡言乱语、无限重复等等
            exists_think_tag = text.find('</think>') > -1
            exists_output_answer_tag = text.find('<output_answer>') > -1
            if not exists_think_tag and not exists_output_answer_tag:
                if len_for_exception_response <= 0:
                    return None, False
                else:
                    return text[ - len_for_exception_response : None ], False
            else:
                # 去掉think内容
                if exists_think_tag:
                    text = text.split('</think>', 1)[1].strip()

                # "</output_answer>"可能没有返回，则自动补全
                if exists_output_answer_tag and text[-9:None] != "</output_answer>":
                    text += "</output_answer>"

                # print("Response-text:", text)
                return text, True
        except APIConnectionError as e:
            print(f"连接失败: {e}")
            print("请检查 base_url 是否配置正确，或者网络是否正常。")
            raise e
        except AuthenticationError as e:
            print(f"鉴权失败: {e}")
            print("请检查 API Key 是否填写正确。")
            raise e
        except APIError as e:
            error = str(e)
            if error.find("inappropriate") > 0:
                return f'<output_answer>Output data may contain inappropriate content</output_answer>'
            else:
                raise e
        except BadRequestError as e:
            # Error code: 400 - Input data may contain inappropriate content.
            error = str(e)
            if error.find("inappropriate") > 0:
                return f'<output_answer>Input data may contain inappropriate content</output_answer>'
            else:
                raise e
        except Exception as e:
            error = str(e)
            if error.find("inappropriate") > 0:
                return f'<output_answer>Output data may contain inappropriate content</output_answer>'
            else:
                raise e


    # 并发请求模型服务（流式）
    async def __batch_request_chat_api_stream(prompts: list, max_tokens=10240, temperature=0.6, enable_thinking=True):
        tasks = []
        for prompt in prompts:
            tasks.append(ChatClient.__request_chat_api_stream(prompt, max_tokens, temperature, enable_thinking))

        return await asyncio.gather(*tasks)


    # 模型服务流式输出
    async def __request_chat_api_stream(prompt: str, max_tokens=10240, temperature=0.6, enable_thinking=True, len_for_exception_response=0):
        client = AsyncOpenAI(
            api_key = serve_api_key,
            base_url = serve_base_url.rstrip('/'),
            timeout = 60*60*24,
        )

        messages = [
            {
                "role": "system",
                "content": "你是一名无所不知无所不能的全能高手，请仔细理解任务描述和任务目标，展开所有联想找到最佳策略和方法，尽你所能生成用户期望的标准答案，返回结果的格式也须符合用户要求。"
            },
            {
                "role": "user",
                "content": str(prompt) if enable_thinking else str(prompt) + '/no_think', # 必须强制转一次字符串
            }
        ]

        try:
            response = await client.chat.completions.create(
                model = serve_model_name,
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
                stream = True,
            )

            # 粗略估计字符串长度限制
            max_len = 3 * max_tokens
            full_len = 0
            full_content = ""
            async with response:
                async for chunk in response:
                    content = None
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        content = chunk.choices[0].delta.reasoning_content
                    elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                    else:
                        pass

                    if content != None:
                        full_content += content
                        # 利用 env:DEBUG_PRINT_STREAM 判断是否实时打印（调试查看模型的输出内容）
                        if debug_print_stream == str(CONST_DEBUG_SHOW_STREAM) or int(debug_print_stream) == CONST_DEBUG_SHOW_STREAM:
                            print(content, end="", flush=True)

                        # 若发现长度异常，则主动终止连接提前截断输出，在这里粗略估计即可
                        full_len += len(content)
                        if full_len > max_len:
                            await client.close()
                            break

            text = full_content.strip()
            # print("Response-full_text:", text)

            # 处理异常数据：长度截断、者胡言乱语、无限重复等等
            exists_think_tag = text.find('</think>') > -1
            exists_output_answer_tag = text.find('<output_answer>') > -1
            if not exists_think_tag and not exists_output_answer_tag:
                if len_for_exception_response <= 0:
                    return None, False
                else:
                    return text[ - len_for_exception_response : None ], False
            else:
                # 去掉think内容
                if exists_think_tag:
                    text = text.split('</think>', 1)[1].strip()

                # "</output_answer>"可能没有返回，则自动补全
                if exists_output_answer_tag and text[ -16 : None ] != "</output_answer>":
                    text += "</output_answer>"

                # print("Response-text:", text)
                return text, True

        except APIConnectionError as e:
            print(f"连接失败: {e}")
            print("请检查 base_url 是否配置正确，或者网络是否正常。")
            raise e
        except AuthenticationError as e:
            print(f"鉴权失败: {e}")
            print("请检查 API Key 是否填写正确。")
            raise e
        except APIError as e:
            error = str(e)
            if error.find("inappropriate") > 0:
                return '<output_answer>Output data may contain inappropriate content</output_answer>'
            else:
                raise e
        except BadRequestError as e:
            # Error code: 400 - Input data may contain inappropriate content.
            error = str(e)
            if error.find("inappropriate") > 0:
                return '<output_answer>Input data may contain inappropriate content</output_answer>'
            else:
                raise e
        except Exception as e:
            error = str(e)
            if error.find("inappropriate") > 0:
                return '<output_answer>Output data may contain inappropriate content</output_answer>'
            else:
                raise e


    # 检测服务是否可用
    def ping_serve():
        url = serve_base_url+'/models'
        try:
            if serve_api_key != '':
                response = requests.get(url, headers={'Authorization': f'Bearer {serve_api_key}'}, timeout=30)  # 设置超时时间为30秒
            else:
                response = requests.get(url, timeout=30)

            if response.status_code == 200:
                print(f"URL {url} is reachable.")
                exsits = serve_model_name in [ row['id'] for row in response.json()['data'] ]
                if exsits:
                    return 'serve reachable and serve_model_name is ok'
                else:
                    raise Exception('serve reachable but serve_model_name not exsits')
            else:
                print(f"URL {url} is not reachable. Status code: {response.status_code}")
                raise Exception('serve not reachable')

        except requests.exceptions.RequestException as e:
            print(f"URL {url} is not reachable. Error: {e}")
            raise e


async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def fetch_url_and_count(urls):
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)

    cnts = []
    for url, content in zip(urls, results):
        cnts.append(f"{url}: {len(content)} bytes")

    return cnts


if __name__=="__main__":
    # print('response:', ChatClient.ping_serve())

    # urls = [
    #     "https://example.com",
    #     "https://python.org"
    # ]

    # cnts = asyncio.run(fetch_url_and_count(urls))
    # for s in cnts:
    #     print(s)

    ChatClient.request_llm_api_stream("请列举Golang中GC的原理和案例")
