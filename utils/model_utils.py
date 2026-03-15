from langchain_openai import ChatOpenAI
from embedding_utils import image_to_base64


multimodal_llm = ChatOpenAI(
    model="Qwen3vl",
    base_url="http://localhost:8010/v1",
    api_key="empty"
)

# test
# print(multimodal_llm.invoke([
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "请生成项目的目录结构"
#             }, {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": image_to_base64("../img/img.png")
#                 }
#             }
#         ]
#     }
# ]).content)
