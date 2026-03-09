import os
import sys

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class RagLogger():
    def __init__(self):
        self.logger = logger
        # 清空所有设置
        self.logger.remove()
        # 添加控制台输出的格式
        self.logger.add(sys.stdout, level='DEBUG',
                        format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 线程名
                               "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                               ":<cyan>{line}</cyan> | "  # 行号
                               "<level>{level}</level>: "  # 等级
                               "<level>{message}</level>",  # 日志内容
                        )

    def get_logger(self):
        return self.logger


log = RagLogger().get_logger()
