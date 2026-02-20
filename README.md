# multimodal-RAG
## dots.ocr-1.5 vllm部署
### 问题
+ 模型下载后的目录名称中不能含有"."，因为目录名需要作为python包名，注入vllm
+ dots.ocr-1.5的模型文件中缺少modeling_dots_ocr_vllm.py文件，需要从dots.ocr的模型文件复制过来
+ 官方推荐的transformers==4.56.1版本与vllm==0.9.1版本会冲突，需要降低版本为transformers==4.53.2
### 模型部署
#### python环境
```shell
conda create -n dotsOCR python=3.12
```
#### pip安装
```shell
pip install vllm==0.9.1
pip install transformers==4.53.2
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn安装
# 参照cuda、pytorch、python版本选择flash-attn安装，从如下地址下载：
# https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
#### vllm服务
```shell
# vllm注入
export hf_model_path=/mnt/e/llm/models/rednote-hilab/dots_ocr_1_5
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from dots_ocr_1_5 import modeling_dots_ocr_vllm' `which vllm`

# 服务启动
CUDA_VISIBLE_DEVICES=0 vllm serve \
/mnt/e/llm/models/rednote-hilab/dots_ocr_1_5 \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.9 \
--chat-template-content-format string \
--served-model-name dots_orc_1_5 \
--trust-remote-code
```
