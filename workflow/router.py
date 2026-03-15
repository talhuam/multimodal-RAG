"""
工作流动态路由函数
"""
import os
import sys
from langgraph.graph import END
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from workflow.workflow_state import MultiModalRAGState


def route_only_image(state: MultiModalRAGState):
    """如果仅有图片，则直接查询知识库，否则进入llm节点，凭借短期记忆回答"""
    if state.get("input_type") == "only_image":
        return "retriever_node"
    return "first_chatbot"


def route_llm_or_retriever(state: MultiModalRAGState):
    """如果上下文检索有结果，则进入llm节点，否则进入retriever节点"""
    if messages := state.get("messages", []):
        tool_message = messages[-1]
    else:
        raise ValueError("No message found in input")
    if not tool_message.content or tool_message.content == "没有找到相关的历史上下文信息。":
        return "retriever_node"
    return 'second_chatbot'


def route_evaluate_node(state: MultiModalRAGState):
    """如果用户仅仅输入图片，则不进行评估（目前RAGAS还不支持多模态评估），其他情况下进入评估节点"""
    if state.get("input_type") == "only_image":
        return END
    return "evaluate_node"


def route_human_node(state: MultiModalRAGState):
    """如果评估后的分值低于0.7，则进入人工介入节点"""
    if state.get("evaluate_score") >= 0.7:
        return END
    return "human_approval"


def route_human_approval_node(state: MultiModalRAGState):
    """如果用户输入的是：approve 则结束，否则进入兜底回复"""
    if state.get("human_answer") == "approve":
        return END
    return "fourth_chatbot"
