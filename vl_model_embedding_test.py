# from langchain_openai import OpenAIEmbeddings
#
# embedding = OpenAIEmbeddings(
#     model="Qwen3vl-Embedding",
#     base_url="http://localhost:8020/v1",
#     api_key="empty",
# )
#
# print(embedding.embed_query("yes, i do"))
import json

import requests
import numpy as np

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
        return ""

payload = {
    "model": "Qwen3vl-Embedding",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_base64("./data/demo.jpeg")
                    }
                }
            ]
        }
    ]
}

response = requests.post("http://localhost:8020/v1/embeddings", json=payload)
image_embedding = response.json().get("data")[0].get("embedding")
print(f"image_embedding: {image_embedding}")


payload = {
    "model": "Qwen3vl-Embedding",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_base64("./data/demo.jpeg")
                    }
                }, {
                    "type": "text",
                    "text": "美女和狗处于一片沙滩上，背景是广阔的海洋和正在落下的夕阳。海浪轻轻拍打着岸边，天空被夕阳染成了温暖的橙黄色"
                }
            ]
        }
    ]
}

response = requests.post("http://localhost:8020/v1/embeddings", json=payload)
mixed_embedding = response.json().get("data")[0].get("embedding")
print(f"mixed_embedding: {mixed_embedding}")


desc = """
这张图片描绘了一个温馨的海滩场景，主要包含以下元素：\n- **人物**：一位年轻女子，她坐在沙滩上，面带微笑，穿着一件蓝白格子的衬衫和深色裤子。\n- **动物**：一只金毛寻回犬，它戴着一个带有彩色图案的背带，正坐着，伸出前爪与女子互动。\n- **互动**：女子和狗正在互相“击掌”或“碰爪”，这是一个充满爱意和默契的互动，展现了人与宠物之间的亲密关系。\n- **环境**：他们身处一片沙滩上，背景是广阔的海洋和正在落下的夕阳。海浪轻轻拍打着岸边，天空被夕阳染成了温暖的橙黄色。\n- **氛围**：整个画面沐浴在温暖的金色阳光中，营造出一种宁静、幸福、放松的氛围，充满了生活中的美好与治愈感。\n总的来说，这是一张充满爱与和谐的户外生活照片，捕捉了人与宠物在自然美景中共享美好时光的瞬间。
""".strip()

payload = {
    "model": "Qwen3vl-Embedding",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": desc
                }
            ]
        }
    ]
}

response = requests.post("http://localhost:8020/v1/embeddings", json=payload)
text_embedding = response.json().get("data")[0].get("embedding")
print(f"text_embedding: {text_embedding}")

desc = "美国攻打伊朗"
payload = {
    "model": "Qwen3vl-Embedding",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": desc
                }
            ]
        }
    ]
}
response = requests.post("http://localhost:8020/v1/embeddings", json=payload)
text_embedding_1 = response.json().get("data")[0].get("embedding")
print(f"text_embedding_1: {text_embedding_1}")


def cosine_similarity(vector_a, vector_b):
    # 转换为 numpy 数组（如果还不是）
    a = np.array(vector_a, dtype=np.float32)
    b = np.array(vector_b, dtype=np.float32)
    # 计算点积
    dot_product = np.dot(a, b)
    # 计算模长
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 避免除零错误
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)


print(cosine_similarity(mixed_embedding, text_embedding))
print(cosine_similarity(image_embedding, text_embedding))
print(cosine_similarity(image_embedding, text_embedding_1))



