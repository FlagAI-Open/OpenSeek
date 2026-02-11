import json
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-file-path', type=str, default="", help="file path", required=True)
  parser.add_argument('--metrics', type=str, default="custom|hellaswag|0,custom|arc:challenge|0,custom|arc:easy|0,custom|piqa|0,custom|mmlu_cloze:_average|0,custom|commonsense_qa|0,custom|trivia_qa|0,custom|winogrande|0,custom|openbookqa|0,custom|gsm8k|5,custom|siqa|0,custom|ceval:_average|0,custom|cmmlu:_average|0", help="", required=False)
  parser.add_argument('--acc-norm', action='store_true', help='Use acc norm.')
  args = parser.parse_args()

  metrics = args.metrics.split(',')
  input_file_path = args.input_file_path

  results = json.loads(open(input_file_path).read())['results']

  outputs = []
  for metric in metrics:
    submetric = 'acc'

    if 'custom|trivia_qa|0' in metric or 'custom|gsm8k|5' in metric:
        submetric = 'qem'
    elif args.acc_norm and 'custom|commonsense_qa|0' not in metric:
        submetric = 'acc_norm'

    if metric not in results:
      outputs.append([f"{metric.replace('custom|','').replace('|0','').replace(':_average','').replace(':','_')}", 0.0])
      continue

    acc_norm = results[metric][submetric] * 100
    outputs.append([f"{metric.replace('custom|','').replace('|0','').replace(':_average','').replace(':','_')}", acc_norm])

  # awk(2)
  outputs[1][0] = 'ARC(Average)'
  outputs[1][1] = round((outputs[1][1]+outputs[2][1])/2, 2)
  del outputs[2]

  # awk(1)
  outputs[0][0] = 'HellaSwag'
  outputs[2][0] = 'PIQA'
  outputs[3][0] = 'MMLU(cloze)'
  outputs[4][0] = 'CommonsenseQA'
  outputs[5][0] = 'TriviaQA'
  outputs[6][0] = 'Winograde'
  outputs[7][0] = 'OpenBookQA'
  outputs[8][0] = 'GSM8K(5-shot)'
  outputs[9][0] = 'SIQA'
  outputs[10][0] = 'CEval'
  outputs[11][0] = 'CMMLU'

  # Rounding
  for i in range(0, len(outputs)):
    outputs[i][1] = round(outputs[i][1], 2)

  print('\n'.join([ x[0]+'\t'+str(x[1]) for x in outputs ]))



