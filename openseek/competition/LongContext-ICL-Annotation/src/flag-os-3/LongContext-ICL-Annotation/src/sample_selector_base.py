import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from abc import ABC, abstractmethod
from main import annotate_sample

class SampleSelectorBase(ABC):
    """
    难例样本筛选通用基类，所有任务的筛选逻辑都继承此类
    子类只需要实现差异部分的配置和逻辑，公用逻辑全部封装
    """
    # -------------------------- 通用常量（所有任务共享） --------------------------
    THREAD_POOL_SIZE = 32

    def __init__(self,
                 dataset_path: str,
                 cache_file_path: str,
                 max_cache_size: int = 100,
                 task_name: str = "untitled_task",
                 task_id = 5,
                 args = None):
        """
        初始化筛选器
        :param dataset_path: 数据集路径
        :param cache_file_path: 缓存文件保存路径
        :param max_cache_size: 最大缓存样本数
        :param task_name: 任务名称，用于打印提示
        """
        # 任务特有配置
        self.dataset_path = dataset_path
        self.cache_file_path = cache_file_path
        self.max_cache_size = max_cache_size
        self.task_name = task_name
        self.task_id = task_id
        self.args = args

        # 线程安全资源
        self.cache_lock = threading.Lock()
        self.stop_event = threading.Event()

        # 运行时变量
        self.cache = []
        self.added_count = [0]
        self.total_samples = 0

    # -------------------------- 抽象方法（子类必须实现的差异逻辑） --------------------------
    @abstractmethod
    def build_prompt(self, sample_input: str) -> str:
        """构造零样本判断用的prompt，每个任务不同"""
        pass

    # -------------------------- 公用方法（所有任务共享） --------------------------
    def load_cache(self) -> list:
        """加载现有缓存池，不存在则初始化空缓存"""
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        if not os.path.exists(self.cache_file_path):
            return []
        with open(self.cache_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    def save_cache(self, cache: list) -> None:
        """保存缓存池到文件，自动去重截断"""
        if len(cache) > self.max_cache_size:
            cache = self.filter_representative_samples(cache, self.max_cache_size)
        with open(self.cache_file_path, 'w', encoding='utf-8') as f:
            for sample_line in cache:
                f.write(sample_line + '\n')

    def filter_representative_samples(self, cache: list, target_size: int) -> list:
        """智能去重筛选，保留最具代表性的样本"""
        if len(cache) <= target_size:
            return cache
        seen_content = set()
        unique_cache = []
        for sample in reversed(cache):  # 优先保留最新的
            input_text = sample.split('<label>')[0].strip('# ').strip()
            if input_text not in seen_content and len(unique_cache) < target_size:
                seen_content.add(input_text)
                unique_cache.append(sample)
        return list(reversed(unique_cache))

    def need_add_to_cache(self, sample_input: str, sample_output: str) -> tuple[bool, str]:
        """零样本判断是否是难例，预测错误则加入缓存"""
        try:
            # 构造prompt（子类实现）
            task_description = self.build_prompt(sample_input)

            # 调用大模型（和method.py参数完全一致）
            prediction = annotate_sample(task_description, sample_input, self.task_id,args = self.args)
            norm_pred,norm_gt = prediction, sample_output

            # 调试打印：前10个样本展示详情
            if len(self.cache) < 10:
                print(f"\n🔍 [调试] 样本：{sample_input[:100]}...")
                print(f"真实标注：{sample_output} (标准化后: {norm_gt})")
                print(f"模型预测：{prediction} (标准化后: {norm_pred})")
                print(f"是否错误：{norm_pred != norm_gt}")
                print("-"*50)

            # 预测错误则作为难例加入
            if norm_pred != norm_gt:
                return True, "零样本预测错误难例"
            return False, ""

        except Exception as e:
            print(f"⚠️  调用大模型出错: {str(e)}")
            return False, ""

    def process_single_sample(self, sample):
        """单个样本处理逻辑，供多线程调用"""
        if self.stop_event.is_set():
            return
        try:
            sample_input = sample['input']
            sample_output = sample['output'][0]

            # 判断是否是难例
            need_add, reason = self.need_add_to_cache(sample_input, sample_output)
            self.added_count[0] += 1

            if need_add and not self.stop_event.is_set():
                with self.cache_lock:
                    # 双重检查防止缓存已满
                    if len(self.cache) >= self.max_cache_size:
                        self.stop_event.set()
                        return

                    # 构造样本行，只保留输入输出，不要额外注释
                    sample_line = f"# {sample_input} <label>{sample_output}</label>"
                    self.cache.append(sample_line)
                    current_size = len(self.cache)

                    # 实时写入
                    self.save_cache(self.cache)

                    # 打印进度
                    print(f"\n✅ [{self.task_name}] 新增第{current_size}个样本，当前缓存大小: {current_size}/{self.max_cache_size}")
                    print(f"📊 进度：已处理{self.added_count[0]}个，完成度：{round(self.added_count[0]/self.total_samples*100, 1)}%")

                    # 达到上限停止
                    if current_size >= self.max_cache_size:
                        self.stop_event.set()
                        print(f"\n🎉 [{self.task_name}] 已达到{self.max_cache_size}个样本上限，提前终止！")

        except Exception as e:
            print(f"⚠️  处理样本出错: {str(e)}")

    def run(self):
        """启动筛选主流程"""
        # 加载现有缓存
        self.cache = self.load_cache()
        current_size = len(self.cache)
        print(f"📂 [{self.task_name}] 加载现有缓存，共 {current_size} 个样本")

        if current_size >= self.max_cache_size:
            print(f"✅ [{self.task_name}] 缓存已满，无需新增")
            return

        # 加载数据集
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        all_samples = dataset['examples']
        self.total_samples = len(all_samples)
        print(f"📊 [{self.task_name}] 加载数据集，共 {self.total_samples} 个样本")
        print(f"🎯 构建目标：新增最多{self.max_cache_size - current_size}个难例样本")
        print(f"🚀 启动{self.THREAD_POOL_SIZE}线程并发处理...")

        # 初始化状态
        self.added_count = [0]
        self.stop_event.clear()

        # 多线程处理
        try:
            with ThreadPoolExecutor(max_workers=self.THREAD_POOL_SIZE) as executor:
                futures = [executor.submit(self.process_single_sample, sample) for sample in all_samples]
                for future in as_completed(futures):
                    if self.stop_event.is_set():
                        for f in futures:
                            f.cancel()
                        break
        except KeyboardInterrupt:
            print(f"\n⏹️  用户手动终止，已保存当前缓存")
            self.stop_event.set()

        # 最终整理
        if len(self.cache) > self.max_cache_size:
            self.cache = self.filter_representative_samples(self.cache, self.max_cache_size)
            self.save_cache(self.cache)
            print(f"🔄 最终筛选后缓存大小: {len(self.cache)}/{self.max_cache_size}")

        print(f"\n🎉 [{self.task_name}] 处理完成！本次新增 {len(self.cache) - current_size} 个样本")
        print(f"💾 缓存文件：{self.cache_file_path}")
