#!/bin/bash
ORIGINAL_DIR=$(pwd)
# Step 1: Prompt user to activate the environment
echo "Please ensure your virtual environment is activated before proceeding."
read -p "Press Enter to continue after activating the environment..."

# Step 2: Download dataset from Hugging Face
echo "Downloading dataset from https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B..."
echo "WARNING: *************** This method is not recommended, we suggest you to download the dataset manually from https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B and change the 'experiment.dataset_base_dir' in configs/OpenSeek-Small-v1-Baseline/config_deepseek_v3_1_4b.yaml with your download path. Then you can commented out the Step 2 and run this script again. ***************"
DATASET_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BAAI/OpenSeek-Pretrain-100B', repo_type='dataset'))")
if [ $? -ne 0 ]; then
    echo "Error: Failed to download dataset."
    exit 1
fi
echo "Dataset downloaded to: $DATASET_PATH"

# Step 3: Clone specific commit of FlagScale repository
echo "Cloning FlagScale repository at commit 10faa30d1635fb8a102a48352d196870841100a0..."
git clone --no-checkout https://github.com/FlagOpen/FlagScale.git
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone repository."
    exit 1
fi
cd FlagScale
git checkout 10faa30d1635fb8a102a48352d196870841100a0
if [ $? -ne 0 ]; then
    echo "Error: Failed to checkout commit."
    exit 1
fi

# Step 4: Run the unpatch script
echo "Running unpatch script..."
python3 tools/patch/unpatch.py --backend Megatron-LM
if [ $? -ne 0 ]; then
    echo "Error: Failed to run unpatch script."
    exit 1
fi

# Step 5: Return to original directory and print completion message
cd "$ORIGINAL_DIR"
echo "Setup complete!"
echo "Then you can run the script from the OpenSeek root directory with the command:
      bash openseek/baseline/run_exp.sh start"
