ulimit -Sn 1000000
INPUT_DIR=/your/local/dataset/dir
WORKING_DIR=/your/output/dir
THREADS=64
TASKS=32
LENGTH_THRESHOLD=200
MIN_DOC_WORDS=35
FILE_TYPE=jsonl

python scripts/load_dataset_local.py \
    --data-folder $INPUT_DIR \
    --working-folder $WORKING_DIR \
    --file-type $FILE_TYPE \
    --threads $THREADS \
    --tasks $TASKS

python scripts/make_suffix_array.py \
    --working_folder $WORKING_DIR 

./target/debug/dedup_dataset self-similar \
    --data-file $WORKING_DIR/es/dataset.big_sequence \
    --length-threshold $LENGTH_THRESHOLD \
    --cache-dir $WORKING_DIR/cache \
    --num-threads $THREADS

./target/debug/dedup_dataset collect \
    --data-file $WORKING_DIR/es/dataset.big_sequence \
    --cache-dir $WORKING_DIR/cache \
    --length-threshold $LENGTH_THRESHOLD \
    > $WORKING_DIR/es/dataset.big_sequence.remove.bytearange

python scripts/remove_dedup.py \
    --working-folder $WORKING_DIR \
    --min-doc-words $MIN_DOC_WORDS \
    --tasks $TASKS \
    --threads $THREADS

