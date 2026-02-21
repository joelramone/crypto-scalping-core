from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "event": record.getMessage(),
        }

        for key in ("event_name", "parameters", "duration", "duration_s", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class EventAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("event_name", msg)
        if "parameters" not in extra:
            extra["parameters"] = {}
        return msg, kwargs


class TimedEvent:
    def __init__(self, logger: EventAdapter, event_name: str, **parameters: Any):
        self._logger = logger
        self._event_name = event_name
        self._parameters = parameters
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        self._logger.info(self._event_name, extra={"event_name": self._event_name, "parameters": self._parameters})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_s = time.perf_counter() - self._start
        level = logging.ERROR if exc_type else logging.INFO
        self._logger.log(
            level,
            f"{self._event_name}_completed",
            extra={
                "event_name": f"{self._event_name}_completed",
                "parameters": self._parameters,
                "duration_s": duration_s,
            },
            exc_info=exc_val,
        )


def configure_logging(default_level: str = "INFO") -> None:
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = JsonFormatter()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> EventAdapter:
    return EventAdapter(logging.getLogger(name), {})


def timed_event(logger: EventAdapter, event_name: str, **parameters: Any) -> TimedEvent:
    return TimedEvent(logger, event_name, **parameters)
