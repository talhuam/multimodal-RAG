import asyncio
import os
import sys
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from workflow.tools import search_context
from workflow.workflow_state import MultiModalRAGState, InvalidInputException
from utils.log_utils import log
from utils.model_utils import multimodal_llm
from workflow.search_node import SearchContextToolNode
from workflow.retrieve_node import retriever_node
from workflow.evaluate_node import evaluate_answer
from workflow.router import route_only_image, route_llm_or_retriever, route_evaluate_node, route_human_node, \
    route_human_approval_node
from workflow.print_messages import pretty_print_messages
from workflow.context_saver import get_milvus_writer
from utils.embedding_utils import image_to_base64


# 工具
tools = [search_context]


# 节点函数
def process_input(state: MultiModalRAGState, config: RunnableConfig):
    """处理用户输入"""
    user_name = config["configurable"].get("user_name", "Alice")
    last_message = state["messages"][-1]
    log.info(f"用户 {user_name} 输入：{last_message}")
    input_type = 'has_text'
    text_content = None
    image_url = None
    # 检查输入类型
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):  # 多模态的消息
            content = last_message.content
            for item in content:
                # 提取文本内容
                if item.get("type") == "text":
                    text_content = item.get("text", None)

                # 提取图片URL
                elif item.get("type") == "image_url":
                    url = item.get("image_url", "").get('url')
                    if url:  # 确保URL有效（是图片的base64格式的字符串 或者 在线url）
                        image_url = url
    else:
        raise InvalidInputException(f"用户输入的消息错误！原始输入：{last_message}")

    if not text_content and image_url:
        input_type = 'only_image'

    # 保存状态
    return {
        "input_type": input_type,
        "user_name": user_name,
        "input_text": text_content,
        "input_image": image_url
    }


# 节点函数，第一次回复（基于当前会话生成回复）
def first_chatbot(state: MultiModalRAGState):
    llm_with_tools = multimodal_llm.bind_tools(tools)
    system_message = SystemMessage(content="""你是一名专精于 Apache Flink 的AI助手。你的核心任务是处理用户关于 Flink 的技术问题。

    # 核心指令（必须严格遵守）：
    1.  **首要规则**：当用户提问涉及 Apache Flink 的任何技术概念、配置、代码或实践时，你**必须且只能**调用 `search_context` 工具来获取信息。
    2.  **禁止行为**：你**严禁**凭借自身内部知识直接回答任何 Flink 技术问题。你的回答必须完全基于工具返回的知识库内容。
    3.  **兜底策略**：如果工具返回了相关信息，请基于这些信息组织答案。如果工具明确返回“未找到相关信息”，你应统一回复：“关于这个问题，我当前的知识库中没有找到确切的资料。”

    # 回答流程（不可更改）：
    用户提问 -> 调用 `search_context` 工具 -> 基于工具返回结果生成答案。
    """)
    return {"messages": [llm_with_tools.invoke([*state["messages"], system_message])]}


# 节点函数，第二次回复（基于检索历史上下文 生成回复, 检索到的历史上下文在ToolMessage里面）
def second_chatbot(state: MultiModalRAGState):
    return {"messages": [multimodal_llm.invoke(state["messages"])]}


# 节点函数，第三次回复（基于检索知识库上下文 生成回复, 检索到的结果在状态里面）
def third_chatbot(state: MultiModalRAGState):
    """处理多模态请求并返回Markdown格式的结果"""
    context_retrieved: list[dict] = state.get('context_retrieved')
    images: list = state.get('images_retrieved')  # 图片路径

    # 处理上下文列表
    count = 0
    context_pieces = []
    for hit in context_retrieved:
        count += 1
        context_pieces.append(f"检索后的内容{count}：\n {hit.get('text')} \n 资料来源：{hit.get('filename')}")

    context = "\n\n".join(context_pieces) if context_pieces else "没有检索到相关的上下文信息。"

    input_text = state.get('input_text')
    input_image = state.get('input_image')

    # 构建系统提示词
    system_prompt = f"""
        请根据用户输入和以下检索到的上下文内容生成响应，如果上下文内容中没有相关答案，请直接说明，不要自己直接输出答案。
        要求：
        1. 响应必须使用Markdown格式
        2. 在响应文字下方显示所有相关图片，图片的路径列表为{images}，使用Markdown图片语法：
        3. 在相关图片下面的最后一行显示上下文引用来源（来源文件名）
        4. 如果用户还输入了图片，请也结合上下文内容，生成文本响应内容。
        5. 如果用户还输入了文本，请结合上下文内容，生成文本响应内容。

        上下文内容：
        {context}
        """

    # 构建用户消息内容
    user_content = []
    if input_text:
        user_content.append({"type": "text", "text": input_text})
    if input_image:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})

    messages = [{
        "role": "system",
        "content": system_prompt
    },{
        "role": "user",
        "content": user_content
    }]

    return {"messages": multimodal_llm.invoke(messages)}


# 审批节点函数
def human_approval(state: MultiModalRAGState):
    log.info('已经进入了人工审批节点')
    log.info(f'当前的状态中的人工审批信息：{state["human_answer"]}')


# 节点函数，第四次回复，兜底策略
def fourth_chatbot(state: MultiModalRAGState):
    """兜底回复"""
    input_text = state.get('input_text')
    return {"messages": [AIMessage(content=f"关于您的问题【{input_text}】，我还没有学会，请稍后再试！")]}


checkpointer = InMemorySaver()  # 生产环境采用redis，存储短期记忆
store = InMemoryStore()  # 生产环境采用mongo或者pg，存储长期记忆

builder = StateGraph(MultiModalRAGState)

# 添加节点
builder.add_node("process_input", process_input)
builder.add_node("first_chatbot", first_chatbot)

search_context_node = SearchContextToolNode(tools=tools)
builder.add_node("search_context", search_context_node)

builder.add_node("retriever_node", retriever_node)
builder.add_node("second_chatbot", second_chatbot)
builder.add_node("third_chatbot", third_chatbot)
builder.add_node("evaluate_node", evaluate_answer)
builder.add_node("human_approval", human_approval)
builder.add_node("fourth_chatbot", fourth_chatbot)

# 添加边
builder.add_edge(START, "process_input")
builder.add_conditional_edges(
    "process_input",
    route_only_image,
    {"retriever_node": "retriever_node", 'first_chatbot': 'first_chatbot'}
)
builder.add_conditional_edges(
    "first_chatbot",
    tools_condition,
    {"tools": "search_context", END: END}
)
builder.add_conditional_edges(
    "search_context",
    route_llm_or_retriever,
    {"retriever_node": "retriever_node", 'second_chatbot': 'second_chatbot'}
)
builder.add_edge("retriever_node", "third_chatbot")
builder.add_conditional_edges(
    "third_chatbot",
    route_evaluate_node,
    {"evaluate_node": "evaluate_node", END: END}
)
builder.add_conditional_edges(
    'evaluate_node',
    route_human_node,
    {"human_approval": "human_approval", END: END}
)
builder.add_conditional_edges(
    'human_approval',
    route_human_approval_node,
    {"fourth_chatbot": "fourth_chatbot", END: END}
)
builder.add_edge("fourth_chatbot", END)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=["human_approval"]
)

# draw graph
# mermaid_code = graph.get_graph().draw_mermaid_png()
# with open("../img/multimodal_rag.png", "wb") as f:
#     f.write(mermaid_code)

session_id = str(uuid.uuid4())

# 配置参数，包含乘客ID和线程ID
config = {
    "configurable": {
        "user_name": "Alice",
        "thread_id": session_id,
    }
}


def update_state(user_input, config):
    if user_input == "approval":
        new_message = "approve"
    else:
        new_message = "rejected"
    graph.update_state(
        config=config,
        values={"human_answer": new_message}
    )


async def execute_graph(user_input:str) -> str:
    current_state = graph.get_state(config)  # 实时的状态（短期上下文）
    if current_state.next:
        # 出现中断，恢复工作流
        update_state(user_input, config)
        # 恢复执行工作流
        async for chunk in graph.astream(None, config, stream_mode="values"):
            pretty_print_messages(chunk)
        return ""
    else:
        message_content = []
        if "&" in user_input:
            # 含 图 和 文
            fields = user_input.split("&")
            text = fields[0]
            image = fields[1]
            if text:
                message_content.append({"type": "text", "text": text})
            if image and os.path.isfile(image):
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_base64(image)
                    }
                })
        elif os.path.isfile(user_input):
            # 只有 图
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_base64(user_input)
                }
            })
        else:
            # 只有 文
            message_content.append({
                "type": "text",
                "text": user_input
            })
        # 调用工作流
        message = HumanMessage(content=message_content)
        async for chunk in graph.astream({'messages': [message]}, config, stream_mode='values'):
            pretty_print_messages(chunk, last_message=True)
    current_state = graph.get_state(config)

    if current_state.next:  # 出现了工作流的中断
        output = ("由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n "
                  "如果认可，请输入“approve”，否则请输入“rejected”，系统将会重新生成回复！")
        return output
    else:
        # 工作流正常完成
        # 异步写入响应到Milvus（把当前工作流执行后的最终结果，保存到上下文的向量数据库中）
        mess = current_state.values.get('messages', [])
        if mess:
            if isinstance(mess[-1], AIMessage):
                log.info(f"开始写入Milvus:")
                # 异步写入Milvus
                task = asyncio.create_task(  # 创建一个多线程异步任务
                    get_milvus_writer().async_insert(
                        context_text=mess[-1].content,
                        user=current_state.values.get('user', 'ZS'),
                        message_type="AIMessage"
                    )
                )

    return ""


async def main():
    # 执行工作流
    while True:
        user_input = input('用户输入(文本和图片用&隔开)：')
        if user_input.lower() in ['exit', 'quit', '退出']:
            break

        res = await execute_graph(user_input)
        if res:
            print('AI: ', res)


if __name__ == '__main__':
    asyncio.run(main())



