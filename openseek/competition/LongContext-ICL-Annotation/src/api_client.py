import os
import re
import json
import time
import requests
import asyncio
import aiohttp

from openai import OpenAI, AsyncOpenAI, APIConnectionError, AuthenticationError, APIError, BadRequestError

from const import CONST_DEBUG_SHOW_STREAM, CONST_DEBUG_SHOW_ANSWER


serve_api_key = os.environ.get('SERVE_API_KEY', '')
serve_base_url = os.environ.get('SERVE_BASE_URL', 'http://localhost:2026/v1')
serve_model_name = os.environ.get('SERVE_MODEL_NAME', 'Qwen/Qwen3-4B')
try:
    debug_print_stream = int(os.environ.get('DEBUG_PRINT_STREAM', 0))
except ValueError:
    debug_print_stream = 0


# 文件目录: {your-project-path}/OpenSeek/openseek/competition/LongContext-ICL-Annotation
LongContext_ICL_Annotation_DIR = os.path.dirname(os.path.dirname(__file__))

llm_visit_history_filepath = os.path.join(LongContext_ICL_Annotation_DIR, 'llm_visit_history.log')

# 记录ICL提取执行进度
def log_llm_visit_history(*args):
    with open(llm_visit_history_filepath, 'a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' :: ' + json.dumps(args)+'\n')


# 使用CHAT模版请求模型服务端
class ChatClient:

    __client = AsyncOpenAI(base_url = serve_base_url.rstrip('/'), api_key = serve_api_key, timeout = 3600)

    # 多个并发请求模型服务（流式）
    @staticmethod
    def batch_request_llm_api_stream(prompts: list, max_tokens=10240, temperature=0.6, enable_thinking=True):
        answers = []
        texts = asyncio.run(ChatClient.__batch_request_chat_api_stream(prompts, max_tokens, temperature, enable_thinking))
        log_llm_visit_history('batch_request_llm_api_stream-p1', '__batch_request_chat_api_stream', 'state', [item[1] for item in texts], 'max_tokens', max_tokens)
        for i, (text, state) in enumerate(texts):
            if debug_print_stream == str(CONST_DEBUG_SHOW_ANSWER) or debug_print_stream == CONST_DEBUG_SHOW_ANSWER:
                print('text, state = ', text, state)

            # 若格式不符合要求，则多请求几次尽量提高成功率
            if state == False:
                time.sleep(3)
                text, state = asyncio.run(ChatClient.__request_chat_api_stream(prompts[i], max_tokens, temperature, enable_thinking, 256))
                log_llm_visit_history('batch_request_llm_api_stream-p2', '__request_chat_api_stream', 'state', state, 'max_tokens', max_tokens)
                if debug_print_stream == str(CONST_DEBUG_SHOW_ANSWER) or debug_print_stream == CONST_DEBUG_SHOW_ANSWER:
                    print('text, state = ', text, state)

            answers.append((text, state))

        return answers


    # 单个请求模型服务（流式）
    @staticmethod
    def request_llm_api_stream(prompt: str, max_tokens=10240, temperature=0.7, enable_thinking=True):
        text, state = asyncio.run(ChatClient.__request_chat_api_stream(prompt, max_tokens, temperature, enable_thinking))
        log_llm_visit_history('request_llm_api_stream-p1', '__request_chat_api_stream', 'state', state, 'max_tokens', max_tokens)
        if debug_print_stream == str(CONST_DEBUG_SHOW_ANSWER) or debug_print_stream == CONST_DEBUG_SHOW_ANSWER:
            print('text, state = ', text, state)

        # 若格式不符合要求，则多请求一次尽量提高成功率
        if state == False:
            time.sleep(3)
            text, state = asyncio.run(ChatClient.__request_chat_api_stream(prompt, max_tokens, temperature, enable_thinking, 256))
            log_llm_visit_history('request_llm_api_stream-p2', '__request_chat_api_stream', 'state', state, 'max_tokens', max_tokens)
            if debug_print_stream == str(CONST_DEBUG_SHOW_ANSWER) or debug_print_stream == CONST_DEBUG_SHOW_ANSWER:
                print('text, state = ', text, state)

        return text, state


    # 并发请求模型服务（流式）
    @staticmethod
    async def __batch_request_chat_api_stream(prompts: list, max_tokens=10240, temperature=0.6, enable_thinking=True):
        tasks = []
        for prompt in prompts:
            tasks.append(ChatClient.__request_chat_api_stream(prompt, max_tokens, temperature, enable_thinking))

        return await asyncio.gather(*tasks)


    # 模型服务流式输出
    @staticmethod
    async def __request_chat_api_stream(prompt: str, max_tokens=10240, temperature=0.6, enable_thinking=True, len_for_exception_response=0):
        messages = [
            {
                "role": "system",
                "content": "你是一名无所不知无所不能的全能高手，请仔细理解任务描述和任务目标，展开所有联想找到最佳策略和方法，尽你所能生成用户期望的最终结果，返回结果的格式也须符合用户使用要求。"
            },
            {
                "role": "user",
                "content": str(prompt) if enable_thinking else str(prompt) + '/no_think', # 必须强制转一次字符串
            }
        ]

        try:
            response = await ChatClient.__client.chat.completions.create(
                model = serve_model_name,
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
                stream = True,
            )

            # 粗略估计字符串长度限制
            max_len = 4 * max_tokens
            full_len = 0
            full_content = ""
            async with response:
                async for chunk in response:
                    content = None
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if getattr(delta, 'reasoning_content', None):
                            content = delta.reasoning_content
                        elif getattr(delta, 'content', None):
                            content = delta.content

                    if content != None:
                        full_content += content
                        # 利用 env:DEBUG_PRINT_STREAM 判断是否实时打印（调试查看模型的输出内容）
                        if debug_print_stream == str(CONST_DEBUG_SHOW_STREAM) or int(debug_print_stream) == CONST_DEBUG_SHOW_STREAM:
                            print(content, end="", flush=True)

                        # 若发现长度异常，则主动终止连接提前截断输出，在这里粗略估计即可
                        full_len += len(content)
                        if full_len > max_len:
                                break

            text = full_content.strip()
            # print("Response-full_text:", text)

            if text.startswith('<think>'):
                idx = text.find('</think>')
                if idx > -1:
                    # 去掉think内容
                    text = text[idx + 8 : None].strip()
                else:
                    # 处理异常数据：胡言乱语、无限重复等等导致输出不完整
                    if len_for_exception_response <= 0:
                        return None, False
                    else:
                        return text[ - len_for_exception_response : None ], False

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
            if "inappropriate" in error:
                return '<output_answer>Output data may contain inappropriate content</output_answer>', False
            else:
                raise e
        except BadRequestError as e:
            # Error code: 400 - Input data may contain inappropriate content.
            error = str(e)
            if "inappropriate" in error:
                return '<output_answer>Input data may contain inappropriate content</output_answer>', False
            else:
                raise e
        except Exception as e:
            error = str(e)
            if "inappropriate" in error:
                return '<output_answer>Output data may contain inappropriate content</output_answer>', False
            else:
                raise e

if __name__=="__main__":
    debug_print_stream = 1
    ChatClient.request_llm_api_stream("请列举Golang中GC的原理和案例")
