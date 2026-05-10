import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    if log_dir is None:
        # Default to PROJECT_ROOT/logs
        script_dir = Path(__file__).parent
        log_dir = str(script_dir.parent / "logs")
    """返回同时写控制台和文件的 Logger。

    文件名格式：<log_dir>/<name>_YYYYMMDD_HHMMSS.log
    日志格式：2026-04-18 21:45:00 [INFO] 消息内容
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(Path(log_dir) / f"{name}_{timestamp}.log")

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（多次调用时）
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Log file: {log_file}")
    return logger
