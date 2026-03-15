from langchain_core.messages import AIMessage
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from evaluate.evaluate import RAGEvaluator
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from workflow.workflow_state import MultiModalRAGState
from utils.model_utils import multimodal_llm
from utils.embedding_utils import embeddings
from utils.log_utils import log

inference_model = LangchainLLMWrapper(multimodal_llm)
embeddings_model = LangchainEmbeddingsWrapper(embeddings)

# 创建RAG评估器
rag_evaluator = RAGEvaluator(inference_model, embeddings_model)


async def evaluate_answer(state: MultiModalRAGState):
    """评估大模型的响应和用户输入之间的相关性"""
    context_retrieved = state.get('context_retrieved')
    input_text = state.get('input_text')
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        answer = last_message.content
    score = await rag_evaluator.evaluate_answer(input_text, context_retrieved, answer)
    log.info(f"RAG Evaluation Score: {score}")
    return {"evaluate_score": float(score)}