
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n flagscale python=3.11.11 -y
conda activate flagscale

pip install --upgrade setuptools

pip --trusted-host pypi.tuna.tsinghua.edu.cn install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r ./requirements/requirements-base.txt
pip install -r ./requirements/requirements-common.txt

pip install deepspeed
pip3 install --no-build-isolation transformer_engine[pytorch]==2.6.0.post1
pip install nvidia-cudnn-frontend

cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
flash_attn_version="2.8.3"
echo "https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/flash_attn-${flash_attn_version}+cu${cu}torch${torch}-cp${cp}-cp${cp}-linux_x86_64.whl"
wget --continue --timeout=60 --no-check-certificate --tries=5 --waitretry=10 https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/flash_attn-${flash_attn_version}+cu${cu}torch${torch}-cp${cp}-cp${cp}-linux_x86_64.whl
flash_attn-${flash_attn_version}+cu${cu}torch${torch}-cp${cp}-cp${cp}-linux_x86_64.whl
# Recommend to download the wheel handly, for example flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64
pip install flash_attn-2.8.3+cu124torch2.6-cp311-cp311-linux_x86_64.whl

# maybe slow, be patient
pip install --no-build-isolation "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"


# Maybe slow too, be patient
pip install -r ./requirements/inference/requirements.txt
pip install vllm==0.8.5
python tools/patch/unpatch.py --backend llama.cpp
python tools/patch/unpatch.py --backend omniinfer
python tools/patch/unpatch.py --backend Megatron-LM

pip install build
pip install  setuptools-scm
pip install "git+https://github.com/state-spaces/mamba.git@v2.2.4"

pip install -r ./requirements/serving/requirements.txt
pip install --no-build-isolation git+https://github.com/FlagOpen/FlagGems.git@release_v1.0.0