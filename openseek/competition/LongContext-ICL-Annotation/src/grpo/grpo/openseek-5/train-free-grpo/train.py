import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List,  Optional
from openai import OpenAI
from tqdm import tqdm
import copy
import dotenv

dotenv.load_dotenv()

TASK = "In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. Label the instances as \"Sad\" or \"Not sad\" based on your judgment. You can get help from hashtags and emojis, but you should not judge only based on them, and should pay attention to tweet's text as well."


class TrainingFreeGRPO:
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None,
                 max_workers: int = 5,
                 temperature: float = 1.0):

        self.client = OpenAI(
            api_key=api_key if api_key else os.getenv('API_KEY'),
            base_url=base_url if base_url else os.getenv('BASE_URL')
        )
        self.model_name = model_name if model_name else os.getenv('MODEL_NAME')
        self.max_workers = max_workers
        self.temperature = temperature

    def get_llm_output(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2048,
            temperature=self.temperature,
            extra_body={
                "thinking": {"type": "disabled"}  # 禁用思考过程，让模型直接输出结果
            },
        )
        return response.choices[0].message.content

    def get_llm_output_batch(self, prompts: List[str]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.get_llm_output, prompts))

    def load_experiences(self, experiences_file: str = 'experiences.json') -> str:

        if os.path.exists(experiences_file):
            with open(experiences_file, 'r', encoding='utf-8') as ef:
                experiences = json.load(ef)
                return experiences
        return {}

    def train(self,
              train_data_file: str = 'train_data.json',
              experiences_file: str = 'experiences.json',
              batch_size: int = 2,
              epochs: int = 3,
              n: int = 4) -> None:

        from prompts import EXPERIENCE_GENERATE_PROMPT, PROBLEM_WITH_EXPERIENCE_PROMPT, BATCH_EXPERIENCE_UPDATE_PROMPT

        train_data = json.load(open(train_data_file, 'r', encoding='utf-8'))[:100]
        # 添加全局进度条
        steps = len(train_data) * epochs // batch_size + 1
        pbar = tqdm(total=steps, desc="Training Progress")
        for epoch in range(epochs):
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                experiences = self.load_experiences(experiences_file)
                prompts = []
                for item in batch:
                    item_prompts = [PROBLEM_WITH_EXPERIENCE_PROMPT.format(
                        task=TASK,
                        problem=item['prompt'],
                        experiences=experiences
                    )] * n
                    prompts.extend(item_prompts)

                llm_outputs = self.get_llm_output_batch(prompts)

                candidate_experiences = copy.deepcopy(experiences)
                to_modify = []
                for idx, item in enumerate(batch):
                    start = idx * n
                    end = start + n
                    trajectory_texts = llm_outputs[start:end]

                    # trajectories = []
                    # for t_idx, traj_text in enumerate(trajectory_texts):
                    #     trajectories.append({
                    #         "id": f"T_{t_idx}",
                    #         "text": traj_text
                    #     })

                    experience_prompt = EXPERIENCE_GENERATE_PROMPT.format(
                        problem=item['prompt'],
                        # trajectories='\n\n'.join([f'ID: {traj["id"]}\nText: {traj["text"]}'
                        #                         for traj in trajectories]),
                        trajectories='\n\n'.join(trajectory_texts),
                        answer=item['answer'],
                        experiences=experiences,
                        max_operations=1
                    )

                    # 生成经验
                    single_sample_experience = self.get_llm_output(experience_prompt)
                    single_sample_experience = single_sample_experience.split('```json')[-1].split('```')[0].strip()

                    try:
                        single_sample_experience = json.loads(single_sample_experience)
                        for exp in single_sample_experience:
                            if exp['option'] == 'add':
                                candidate_experiences[str(len(candidate_experiences) + 1)] = exp['experience']
                            elif exp['option'] == 'modify':
                                if str(exp['modified_from']) in candidate_experiences:
                                    to_modify.append(exp)

                    except Exception as e:
                        print(f"Error parsing experience output: {e}")
                        continue
                update_experiences = self.get_llm_output(BATCH_EXPERIENCE_UPDATE_PROMPT.format(
                    experiences=candidate_experiences,
                    updates=to_modify
                ))
                update_experiences = update_experiences.split('```json')[-1].split('```')[0].strip()
                new_experiences = copy.deepcopy(candidate_experiences)
                pbar.update(1)
                try:
                    update_experiences = json.loads(update_experiences)
                    max_id = max([int(k) for k in new_experiences.keys()], default=0)
                    for exp in update_experiences:
                        if exp['option'] == 'modify':
                            if str(exp['modified_from']) in new_experiences:
                                new_experiences[str(exp['modified_from'])] = exp['experience']
                            elif exp['option'] == 'merge':
                                to_merge = exp.get('merged_from', [])
                                for exp_id in to_merge:
                                    if str(exp_id) in new_experiences:
                                        del new_experiences[str(exp_id)]

                                new_experiences[str(max_id + 1)] = exp['experience']
                                max_id += 1

                    # 重新编号
                    new_experiences = {str(idx + 1): exp for idx, exp in enumerate(new_experiences.values())}

                    with open(experiences_file, 'w', encoding='utf-8') as ef:
                        json.dump(new_experiences, ef, ensure_ascii=False, indent=4)

                except Exception as e:
                    print(f"Error parsing final experience updates: {e}")
                    continue


def main():
    trainer = TrainingFreeGRPO(
        # max_workers=1,
        max_workers=6,
        temperature=1.0
    )

    # 生成经验
    trainer.train(
        train_data_file='train_data.json',
        experiences_file='experiences.json',
        batch_size=8,
        epochs=3,
        n=4
        # batch_size=1,
        # epochs=3,
        # n=1
    )


if __name__ == "__main__":
    main()

    # import json
    # import datasets
    # path = r"E:\dataset\PrimeIntellect\AIME-24\data"
    # self_dataset = datasets.load_dataset(path)["train"]
    # # print(self_dataset)
    #
    # train_data = []
    # for i in range(len(self_dataset)):
    #     data_info = self_dataset[i]
    #     train_data.append({
    #         "prompt": data_info["prompt"],
    #         "answer": json.loads(data_info["verification_info"])["ground_truth"],
    #     })
    #
    # with open("train_data.json", 'w', encoding='utf-8') as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)
