import logging
from pathlib import Path

import structlog


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.WriteLoggerFactory(
            file=path.open("a", encoding="utf-8")
        ),
    )
