import os
import sys
from typing import Dict, List

from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRelevance, ResponseRelevancy, LLMContextPrecisionWithReference, \
    LLMContextPrecisionWithoutReference

from milvus.milvus_retriever import MilvusRetriever
from utils.log_utils import log

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.model_utils import multimodal_llm
from utils.embedding_utils import embeddings
from milvus.create_milvus_collection import DOC_COLLECTION_NAME, client


def generate_answer(question: str, contexts: List[Dict]) -> str:
    context_str = "\n\n".join([f"上下文 {i + 1}: {context['text']}" for i, context in enumerate(contexts)])
    prompt = f"""
    你是一个AI助手，需要根据提供的上下文回答用户的问题。请确保你的回答基于提供的上下文，不要添加额外信息。

    用户问题: {question}

    检索到的上下文:
    {context_str}

    请基于以上上下文回答用户问题。
    """
    res = multimodal_llm.invoke(prompt)
    return res.content


class RAGEvaluator:
    def __init__(self, inference_model, embedding_model) -> None:
        self.inference_model = inference_model
        self.embedding_model = embedding_model

    async def evaluate_context(self, question: str, contexts: List[Dict]) -> float:
        """
        上下文相关性评估: 检索到的上下文（块或段落）是否与用户输入相关
        """
        contexts = [context['text'] for context in contexts]
        # SingleTurnSample用于表示单轮对话的评估样本
        sample = SingleTurnSample(user_input=question, retrieved_contexts=contexts)
        scorer = ContextRelevance(llm=self.inference_model)
        return await scorer.single_turn_ascore(sample)

    async def evaluate_answer(self, question: str, contexts: List[Dict], response: str) -> float:
        """
        评估生成的答案质量
        """
        contexts = [context['text'] for context in contexts]
        # SingleTurnSample用于表示单轮对话的评估样本
        sample = SingleTurnSample(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=contexts,  # 检索到的上下文
            response=response,  # 生成的答案
        )
        log.info(f"开始评估答案质量, 评估样本为：{sample}")
        scorer = ResponseRelevancy(llm=self.inference_model, embeddings=self.embedding_model)
        return await scorer.single_turn_ascore(sample)

    async def evaluate_metrics(self, question: str, contexts: List[Dict], response: str, reference: str = None):
        """
        评估RAG模型
        Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            response: LLM生成的答案
            reference: 可选，参考答案 (用于评估的基准答案，通常为已知的正确答案)
        """
        # 1. 创建评估样本 (SingleTurnSample)
        sample = SingleTurnSample(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=[context['text'] for context in contexts],  # 检索到的上下文
            response=response,  # 生成的答案
            reference=reference  # 参考答案 (用于需要参考答案的指标)
        )

        # 2. 初始化评估指标
        if reference:
            # 如果有参考答案，则初始化指标为LLMContextPrecisionWithReference
            context_precision = LLMContextPrecisionWithReference(llm=self.inference_model)
        else:
            # 如果没有参考答案，则初始化指标为LLMContextPrecisionWithoutReference
            context_precision = LLMContextPrecisionWithoutReference(llm=self.inference_model)

        # 3、执行评估指标得到结果
        context_precision_score = await context_precision.single_turn_ascore(sample)
        print(f"上下文精确度指标的 Score: {context_precision_score}")


async def main():
    inference_model = LangchainLLMWrapper(multimodal_llm)
    embedding_model = LangchainEmbeddingsWrapper(embeddings)

    # 创建RAG评估器
    rag_evaluator = RAGEvaluator(inference_model, embedding_model)

    question = "有界流和无界流有什么区别？"
    # 检索上下文 (从您的Milvus知识库获取)
    m_re = MilvusRetriever(DOC_COLLECTION_NAME, client)
    contexts = m_re.retrieve(question)

    generated_answer = generate_answer(question, contexts)
    print(f"生成的答案: {generated_answer}")

    await rag_evaluator.evaluate_metrics(question, contexts, generated_answer)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

    