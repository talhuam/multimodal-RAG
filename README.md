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

## neo4j部署
版本选择：5.26.8
下载地址：<a href="https://neo4j.ac.cn/deployment-center">neo4j下载地址</a>
### 安装
```shell
sudo apt install cypher-shell_5.26.8_all.deb
sudo apt install neo4j_5.26.8_all.deb
```
### 编辑neo4j配置
监听所有ip请求，默认localhost
```shell
 server.default_listen_address=0.0.0.0
```
### 启动服务
启动服务后，访问web ui:  
http://localhost:7474
```shell
# 设置初始密码，最开始密码默认是neo4j
neo4j-admin dbms set-initial-password neo4j123
# 启动服务
systemctl start neo4j
```
### 安装apoc
apoc是neo4j的扩展库，为neo4j提供了大量的官方Cypher语法本身不具备的，增强型的过程和函数。在多模态RAG的应用中，能够提供更丰富的schema，让大模型写出更好的cypher sql查询neo4j。
apoc版本需要与neo4j对应，下载地址：<a href="https://github.com/neo4j/apoc/releases/download/5.26.8/apoc-5.26.8-core.jar">apoc下载地址</a>
```shell
# 将下载后的jar包放入到
mv apoc-5.26.8-core.jar /var/lib/neo4j/plugins
# 重启neo4j
systemctl restart neo4j
```

## milvus部署
milvus部署方式有多种：
+ milvus lite是一个python库，只需要指定数据库保存目录即可，适用于几百万个向量的数据规模，**该方式不支持windows环境**
```shell
pip install -U pymilvus 
```
+ standalone，可以通过docker compose部署，也能通过deb或者rpm包部署成系统服务，适用于亿级个向量的数据规模。**本项目采用该方式部署**
+ distributed，需要部署在k8s，适用于亿级到数百亿级个向量的数据规模
> 1. standalone和distributed部署方式可参见<a href="https://milvus.io/docs/zh">milvus官网</a>  
> 2. 图形化界面可采用<a href="https://github.com/zilliztech/attu">Attu</a>
### milvus docker方式安装
+ 下载docker compose配置文件 和 milvus配置文件
```shell
wget https://github.com/milvus-io/milvus/releases/tag/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
# 配置文件
cd /etc
wget https://raw.githubusercontent.com/milvus-io/milvus/v2.6.11/configs/milvus.yaml
```
+ 编辑配置文件，启用安全认证
<img src="img/milvus_security_config.png">
+ 修改docker-compose配置，应用配置文件  
```shell
# 增加一行配置
/etc/milvus.yaml:/milvus/configs/milvus.yaml
```
<img src="img/milvus_docker_compose.png">
+ docker compose启动
```shell
# 启动
docker-compose up -d
# 其他命令
docker-compose down # 停止
```

### 图像化界面Attu安装
```shell
# 替换milvus ip为具体的milvus向量数据库的host
docker run -d -p 8000:3000 -e MILVUS_URL=<milvus ip>:19530 zilliz/attu:v2.6
```
打开web ui:
http://localhost:8000
<img src="img/Attu.png">
### milvus相关
#### 支持的字段
+ 主键：可指定int或者varchar，如果指定auto_id，int型则自增，varchar型则随机生成一个字符串
+ 密集向量：用于语义匹配
+ 稀疏向量：用于全文检索，需要指定**分词器**，中文用jieba，英文则指定standard
+ 标量：用于过滤和范围检索

#### milvus创建collection模版
参见：create_milvus_collection.py