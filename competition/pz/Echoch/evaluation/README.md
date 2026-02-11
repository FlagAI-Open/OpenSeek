# **Evaluation**

This directory contains the necessary resources and scripts for evaluating various language models against a diverse set of benchmarks. Our evaluation process leverages established open-source frameworks and includes a novel adversarial evaluation method for reasoning capabilities.

## **Evaluation Frameworks**

We utilize the following GitHub repositories for our model evaluations:

* **lm-evaluation-harness**: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)  
* **lighteval**: [https://github.com/huggingface/lighteval](https://github.com/huggingface/lighteval)  
* **reflection** (for adversarial evaluation): [https://github.com/Essential-AI/reflection](https://github.com/Essential-AI/reflection)
* **qwen_eval**: [https://github.com/QwenLM/Qwen2.5-Math]

## **Evaluated Datasets and Methods**

### **1\. Using lm-evaluation-harness**

We evaluated models on the following datasets using lm-evaluation-harness:

* hellaswag  
* truthfulqa  
* winogrande  
* commonsense\_qa  
* piqa  
* openbookqa  
* boolq  
* arc\_easy  
* arc\_challenge  
* mmlu  
* gsm8k  
* minerva\_math  
* ceval-valid  
* cmmlu

Simple usage:

```shell
bash lm_eval/run_eval.sh path/to/model
```

vllm usage:
```shell
bash run_eval_vllm.sh path/to/model
```

### **2\. Using lighteval (Customized)**

We customized lighteval for evaluating models on these datasets:

* hellaswag  
* winogrande  
* piqa  
* siqa  
* openbookqa  
* arc:easy  
* arc:challenge  
* commonsense\_qa  
* mmlu\_pro\_cloze  
* mmlu\_pro\_mc  
* boolq  
* trivia\_qa  
* gsm8k

Simple usage:

```shell
bash lighteval/run_eval.sh path/to/model
```

### **3\. Adversarial Reasoning Evaluation**

We conducted adversarial evaluations using an adapted method on the following reasoning datasets:

* cruxeval\_i\_adv  
* cruxeval\_o\_adv  
* gsm8k\_adv  
* gsm8k-platinum\_adv

Simple usage:

```shell
bash adv-reasoning-eval/run_eval.sh path/to/model
```

Our approach is inspired by the adversarial CoT framework from [https://github.com/Essential-AI/reflection](https://github.com/Essential-AI/reflection). Given the relatively small scale of our evaluated model, which lacks emergent CoT generation capabilities, we adapted the evaluation. For each test sample containing both a correct and an incorrect Chain-of-Thought (CoT), we measure the model's perplexity (PPL) on both CoTs. A sample is considered passed if the model assigns a lower PPL to the correct CoT compared to the incorrect one. The final score for a dataset is the proportion of samples that passed this PPL criterion. Following [https://github.com/Essential-AI/reflection](https://github.com/Essential-AI/reflection), we report the pre-training compute for each data point as 6nT, where n and T are the number of parameters and training tokens, respectively.

## **Additional Content**

* The adv-reasoning-eval directory contains plotting code. This includes scatter plots illustrating the performance changes of models trained with CoT (Chain-of-Thought) data, as well as comparative scatter plots showing model performance with and without CoT training.  
* The imgs directory stores various charts and figures representing our evaluation results.  
* Each sub-directory within evaluation typically includes run_eval.sh files, which provide simple usage examples for running the evaluations. Please ensure that all necessary libraries and dependencies are installed before attempting to run these scripts.