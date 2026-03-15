import asyncio
from langchain_core.messages import ToolMessage

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.log_utils import log


class SearchContextToolNode:
    """自定义类，来执行搜索上下文工具
    自定义是为了替代：由LangGraph框架自带的ToolNode（有大模型动态传参 来调用工具）
    自定义采用自定义传参
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs: dict):
        # := 先赋值再判断
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []

        # 并行执行所有工具调用
        tasks = []
        for tool_call in message.tool_calls:
            if tool_call.get("args") and 'query' in tool_call["args"]:
                query = tool_call["args"]["query"]
                log.info(f"开始从上下文中检索：{query}")
            else:
                query = inputs.get('input_text')

            # 使用异步调用
            task = self.tools_by_name[tool_call["name"]].ainvoke(
                {'query': query, 'user_name': inputs.get('user_name', 'Alice')}
            )
            tasks.append((tool_call, task))

        # 等待所有异步调用完成
        tool_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for (tool_call, _), tool_result in zip(tasks, tool_results):
            if isinstance(tool_result, Exception):
                # 错误处理
                tool_result = f"工具执行错误: {str(tool_result)}"

            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}