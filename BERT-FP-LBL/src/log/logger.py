import logging


class Logger:
    def __init__(self, config):
        """创建日志器并设置日志输出的最低等级(CRITICAL > ERROR > WARNING > INFO > DEBUG)"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)
        self.config = config

    def create_stream_handler(self, level=logging.INFO):
        """创建控制台处理器"""
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        # 把格式传给处理器
        stream_handler.setFormatter(self.get_formatter()[0])
        return stream_handler

    def create_file_handler(self, level=logging.INFO):
        """创建文件处理器"""
        file_handler = logging.FileHandler(self.config.log_file, mode=self.config.log_file_mode, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(self.get_formatter()[1])
        return file_handler

    @staticmethod
    def get_formatter(stream_fmt='%(asctime)s %(message)s', file_fmt='%(asctime)s %(message)s'):
        """创建格式器"""
        stream_formatter = logging.Formatter(fmt=stream_fmt)
        file_formatter = logging.Formatter(fmt=file_fmt)
        return stream_formatter, file_formatter

    def create_logger(self):
        """日志器添加控制台处理器和文件处理器"""
        self.logger.addHandler(self.create_stream_handler())
        self.logger.addHandler(self.create_file_handler())
        return self.logger
