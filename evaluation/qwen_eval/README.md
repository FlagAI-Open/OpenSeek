## Requirements
You can install the required packages with the following command:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
```


## Usage
### Generation & Evaluation

To run the evaluation:

1. **Configure the prompt type**: Add your custom prompt type in `utils.py` and specify it in `sh/run_evaluate.sh`

2. **Set the model path**: Update the `MODEL_NAME_OR_PATH` variable in `sh/run_evaluate.sh` with your model's path

3. **Run evaluation**: Execute the following command to generate predictions and evaluate results:
   ```bash
   bash sh/run_evaluate.sh
   ```

### Leaderboard
```bash
BASE_DIR=./eval_example/

python evaluate_final.py --eval_path $BASE_DIR
```
BASE_DIR is the directory that contains different datasets folders. The structure of the directory is as follows:
```bash
BASE_DIR
├── dataset1
│   ├── example.jsonl
├── dataset2
│   ├── example.jsonl
├── dataset3
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