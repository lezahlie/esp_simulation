from os import getppid, path, makedirs
import logging
from sys import exit, stderr, stdout
from inspect import stack
import pandas as pd

global_logger = None
global_logger_handler = None

class LoggerHandler:
    def __init__(self, logger):
        self.logger = logger

    def _log_message(self, log_func, msg, stack_level=3):
        if stack_level:
            frame=stack()[stack_level]
            filename = path.basename(frame.filename)
            function_name = frame.function
            lineno = frame.lineno
            log_func(f"[{filename}:{function_name}:{lineno}] || {msg}")
        else:
            log_func(msg)

    def critical(self, msg: str = "unknown critical msg"):
        self._log_message(self.logger.critical, msg)
        exit(-1)

    def error(self, msg: str = "unknown error msg"):
        self._log_message(self.logger.error, msg)
        exit(-1)

    def warning(self, msg: str = "unknown warning msg"):
        self._log_message(self.logger.warning, msg)

    def info(self, msg: str = "unknown info msg"):
        self._log_message(self.logger.info, msg)

    def debug(self, msg: str = "unknown debug msg"):
        self._log_message(self.logger.debug, msg)


def setup_logger(program_file, log_stdout=False, log_stderr=True):
    global global_logger
    global global_logger_handler

    if global_logger is not None:
        return global_logger

    module_name = path.basename(program_file).replace(".py", "")
    
    unique_id = f"{module_name}_ppid{getppid()}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    log_file = f"{unique_id}.log"
    makedirs("logs", exist_ok=True)
    logs_path = f"logs/{log_file}"

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(logs_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] || [%(levelname)s] || [%(filename)s:%(funcName)s:%(lineno)d] || %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if log_stdout:
        stdout_handler = logging.StreamHandler(stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    if log_stderr:
        stderr_handler = logging.StreamHandler(stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    
    global_logger = logger
    global_logger_handler = LoggerHandler(global_logger)
    return logger


def set_logger_level(level:int=20):
    global global_logger
    if level in logging._levelToName:
        global_logger.setLevel(level)
        for handler in global_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
            #global_logger.info(f"Handler: {handler}, Level: {handler.level}")
        global_logger.info(f"Logger level set to {logging._levelToName[level]}")