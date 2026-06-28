import time
import functools
import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, Callable, Dict

# Context variable to accumulate execution performance metrics per request
request_metrics: ContextVar[Dict[str, Any]] = ContextVar("request_metrics", default={})

class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line structured JSON.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": f"{record.filename}:{record.lineno}"
        }
        
        # Merge extra payload properties if attached
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_data.update(record.extra_data)
            
        return json.dumps(log_data)

def setup_json_logger(name: str = "", level: str = "INFO") -> logging.Logger:
    """
    Configures a logger to output structured JSON to standard output.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = JSONFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def time_it(name: str = None):
    """
    Decorator that measures the execution duration of a sync or async function,
    saving the result into the request_metrics context variable.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        metric_key = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = request_metrics.get()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                # Use dict copy to ensure contextvar mutation visibility
                updated_metrics = dict(metrics)
                updated_metrics[f"{metric_key}_latency_sec"] = duration
                request_metrics.set(updated_metrics)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = request_metrics.get()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                updated_metrics = dict(metrics)
                updated_metrics[f"{metric_key}_latency_sec"] = duration
                request_metrics.set(updated_metrics)

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
