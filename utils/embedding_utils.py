"""
多模态表征(embedding)，支持：
1、文本表征
2、图片表征
3、文本+图片混合表征
"""
import os
import sys
import requests
from langchain_openai import OpenAIEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.log_utils import log


# 主要用于语义切割 和 评估模块
embeddings = OpenAIEmbeddings(
    model="Qwen3-Embedding",
    base_url="http://localhost:8000/v1",
    api_key="empty"
)


class EmbeddingException(Exception):
    pass


def image_to_base64(img: str) -> str:
    """将图片转换为base64编码"""
    try:
        import base64, mimetypes
        # 猜测文件MIME类型
        mime = mimetypes.guess_type(img)[0] or "image/png"
        # 读取文件并编码为base64
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 构建data URI格式
        api_img = f"data:{mime};base64,{b64}"
        # store 用原路径或 basename 或 URL 原值，这里存原字符串
        return api_img
    except Exception as e:
        log.error(f"[图片] 本地文件转 base64 失败：{e}")
        log.exception(e)
        return ""


def vl_embed(text: str = None, image: str = None, max_try_times: int = 3) -> list:
    content = []
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    if image:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_base64(image)
            }
        })

    payload = {
        "model": "Qwen3vl-Embedding",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    idx = 0
    while idx < max_try_times:
        idx += 1
        try:
            response = requests.post("http://localhost:8020/v1/embeddings", json=payload)
            embedding = response.json().get("data")[0].get("embedding")
            if embedding:
                return embedding
            else:
                continue
        except Exception as e:
            log.error(f"获取embedding失败，错误：{e}，重试中[{idx}]...")
            continue

    raise EmbeddingException(f"获取embedding重试{max_try_times}次后仍然失败！")


# test
# if __name__ == '__main__':
#     print(embed(image="../data/images/0b149c7d91a4cd2e8ca615caf8605cf8.png"))