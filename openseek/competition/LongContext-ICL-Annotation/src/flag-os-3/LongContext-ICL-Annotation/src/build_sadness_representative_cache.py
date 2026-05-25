import re
from sample_selector_base import SampleSelectorBase
from config import TASK_FILES, TASK5_CACHE_PATH

class Task5SadnessSelector(SampleSelectorBase):
    """任务5：推文悲伤情绪检测难例筛选器"""

    def __init__(self,args = None):
        super().__init__(
            dataset_path=TASK_FILES[5],
            cache_file_path=TASK5_CACHE_PATH,
            max_cache_size=1000,
            task_name="task5_sadness_detection",
            task_id = 5,
            args = args
        )

    def build_prompt(self, sample_input: str) -> str:
        """构造task5专用的prompt"""
        task_description = "In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. Label the instances as \"Sad\" or \"Not sad\" based on your judgment. You can get help from hashtags and emojis, but you should not judge only based on them, and should pay attention to tweet's text as well."
        return task_description

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str,
                        default='/share/project/wuhaiming/spaces/data_agent/OpenSeek-main/openseek/competition/LongContext-ICL-Annotation/src/Qwen3-4B')
    args = parser.parse_args()
    selector = Task5SadnessSelector(args = args)
    selector.run()
