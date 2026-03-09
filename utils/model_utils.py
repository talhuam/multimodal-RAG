from langchain_openai import ChatOpenAI


multimodal_llm = ChatOpenAI(
    model="Qwen3vl",
    base_url="http://localhost:8010/v1",
    api_key="empty"
)

# test
# print(multimodal_llm.invoke([
#     {
#         "role": "user",
#         "content": "你是谁"
#     }
# ]).content)
