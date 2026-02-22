import logging
import sys
import time
import uuid
from pythonjsonlogger import jsonlogger
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

def setup_logging(log_level="INFO"):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Add request_id to context for logging (simple way)
        logging.info(f"Request started", extra={"request_id": request_id, "path": request.url.path})
        
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        logging.info(f"Request finished", extra={
            "request_id": request_id,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": process_time
        })
        
        return response
