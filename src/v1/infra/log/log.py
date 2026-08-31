import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
import structlog


class LogLevel(IntEnum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING

@dataclass
class LogConfig:
    path: Path
    log_level: LogLevel= field(default=logging.NOTSET)


def configure_logging(config: LogConfig) -> None:
    path = config.path
    path.parent.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(config.log_level),
        logger_factory=structlog.WriteLoggerFactory(
            file=path.open("a", encoding="utf-8")
        ),
    )
