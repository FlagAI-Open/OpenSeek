pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python3.11  -m spacy download en_core_web_sm
pip install -e nanobot -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

__MODEL_NAME__=Qwen3-4B-ascend-flagos
__WORK_DIR__=`pwd`
__MODEL_URL__='http://127.0.0.1:9010/v1/'
__MODEL_PATH__='/FlagRelease/Qwen3-4B-FlagOS-Ascend/'

sed -e "s#__MODEL_NAME__#$__MODEL_NAME__#" -e "s#__WORK_DIR__#$__WORK_DIR__#" -e "s#__MODEL_URL__#$__MODEL_URL__#"  config/config.json__ >config/config.json
sed  "s#__MODEL_PATH__#$__MODEL_PATH__#" run.sh__ >run.sh 
