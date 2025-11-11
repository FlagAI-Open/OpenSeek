import re
import argparse
import time
import os
import jsonlines
from tqdm import tqdm
from loguru import logger
from config import prompts, models, patterns
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
from openai import OpenAI


def generate_response(input_queue, output_queue, prompt, pattern, pbar):
    client: OpenAI = models[model]
    counter = [0, 0]

    while True:
        obj = input_queue.get()
        if obj is None:
            input_queue.put(obj)
            break
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': obj[text_key][:max_length]}
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=0.7,
                top_p=0.95,
            )
        except Exception as e:
            with lock:
                pbar.display(f"Error {e.__class__} occurs when generate response for '{obj[text_key][:10]}...'")
            obj['response'] = e
            obj[label_key] = -1
            counter[1] += 1
        else:
            score = re.search(pattern, response.choices[0].message.content)
            score = int(score.group(1)) if score else -1
            obj['response'] = response.choices[0].message.content
            obj[label_key] = score 
            if score == -1:
                counter[1] += 1
            else:
                counter[0] += 1
        output_queue.put(obj)

        with lock:
            pbar.update(1)

    return counter


def main():
    # 0. init state
    log_path = os.path.join(os.path.dirname(output_path), f"log-{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    error_path = os.path.join(os.path.dirname(output_path), f"error-{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
    logger.add(sink=log_path, level='DEBUG')

    # 1. load dataset
    logger.info(f"Loading dataset from {input_path}")
    input_queue = Queue()
    total_lines = 0
    with jsonlines.open(input_path, 'r') as f:
        for line in f:
            input_queue.put(line)
            total_lines += 1
    input_queue.put(None)
    logger.info(f"Total lines in dataset: {total_lines}")


    # 2. LLM request
    logger.info(f"LLM request with {threads} threads.")
    pbar = tqdm(total=total_lines)
    output_queue = Queue()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(generate_response, input_queue, output_queue, prompts[prompt], patterns[prompt], pbar) for _ in range(threads)]

        stat = [0, 0]
        for future in as_completed(futures):
            counter = future.result()
            stat[0] += counter[0]
            stat[1] += counter[1]
            pbar.display(f'One thread has ended, {counter[0]} succeeded, {counter[1]} failed.')

    output_queue.put(None)
    pbar.close()
    logger.info(f'All threads finished, {stat[0]} successful, {stat[1]} failed.')

    # 3. dump dataset
    if stat[0] > 0:
        logger.info(f"dump {stat[0]} successful results to {output_path}")
        sf = jsonlines.open(output_path, 'w')
    if stat[1] > 0:
        logger.info(f"dump {stat[1]} failed results to {error_path}")
        ef = jsonlines.open(error_path, 'w')

    while True:
        obj = output_queue.get()
        if obj is None:
            break
        if obj[label_key] == -1:
            ef.write(obj)
        else:
            sf.write(obj)

    logger.info(f'done.')
    
    if stat[0] > 0:
        sf.close()
    if stat[1] > 0:
        ef.close()
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default='default')
    parser.add_argument("--model", type=str, choices=models.keys(), required=True)
    parser.add_argument("--prompt", type=str, choices=prompts.keys(), required=True)

    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument("--threads", type=int, default=8)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model = args.model 
    prompt = args.prompt
    max_length = args.max_length
    text_key = args.text_key
    label_key = args.label_key
    threads = args.threads

    if output_path == 'default':
        output_path = f"./results/output-{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
    lock = Lock()

    main()
