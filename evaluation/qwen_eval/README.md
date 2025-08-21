## Requirements
You can install the required packages with the following command:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
```


## Usage
```bash
BASE_DIR=./eval_example/

python evaluate_final.py --eval_path $BASE_DIR
```
BASE_DIR is the directory that contains different datasets folders. The structure of the directory is as follows:
```bash
BASE_DIR
├── math500
│   ├── example.jsonl
├── minerva_math
│   ├── example.jsonl
├── olympiadbench
│   ├── example.jsonl
```

The final output file will be saved in the BASE_DIR folder. And the content of the file will be as follows:
```
{
    "dataset_acc": {
        "math500": 3.4,
        "minerva_math": 3.6765,
        "olympiadbench": 2.963,
        "gsm8k": 3.6391
    },
    "final_acc": 3.41965
}
```