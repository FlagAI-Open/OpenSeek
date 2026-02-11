set -u
  file=$1
set +u
python get_results_v3.py \
  --input-file-path $file \
  --acc-norm > tmp.txt
cat tmp.txt
cut -f 2 tmp.txt | cut -f 1
