cd $FlagScale_HOME/tools/checkpoint

set -u
  expname=$1
  version=$2
set +u

checkpoints=$FlagScale_HOME/$expname/checkpoints
loaddir=$Checkpoints_HOME/$expname
mkdir -p $loaddir
echo $version > $loaddir/latest_checkpointed_iteration.txt
VERSION=$(awk '{printf("%07d", $0)}' $loaddir/latest_checkpointed_iteration.txt)
echo "Processing ...", $VERSION
rm $loaddir/iter_${VERSION}

## To avoid affecting ongoing training tasks that are saving models, create a symbolic link in a new directory.
ln -s $checkpoints/iter_${VERSION} $loaddir/iter_${VERSION}
if [ $? -ne 0 ];then
  echo "Error ...", $VERSION
  exit
fi

ulimit -n 1048576
python3 convert.py \
    --model-type deepseek_v3 \
    --loader mcore \
    --saver transformers \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151851 \
    --load-dir $loaddir --save-dir $loaddir/iter_${VERSION}_hf
