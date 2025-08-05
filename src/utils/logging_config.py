"""
Logging configuration for the asteroid mining classification system.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
import sys

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('asteroid_mining')
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    log_file = log_path / f"asteroid_mining_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Performance log handler
    perf_file = log_path / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(detailed_formatter)
    
    # Create performance logger
    perf_logger = logging.getLogger('asteroid_mining.performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    logger.info(f"Logging initialized - Level: {log_level}, Directory: {log_dir}")
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance for a specific module.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'asteroid_mining.{name}')
    return logging.getLogger('asteroid_mining')

def log_performance(func_name: str, execution_time: float, **kwargs):
    """
    Log performance metrics.
    
    Args:
        func_name: Name of the function
        execution_time: Execution time in seconds
        **kwargs: Additional metrics to log
    """
    perf_logger = logging.getLogger('asteroid_mining.performance')
    
    metrics = {
        'function': func_name,
        'execution_time': execution_time,
        **kwargs
    }
    
    metrics_str = ', '.join([f'{k}={v}' for k, v in metrics.items()])
    perf_logger.info(f"PERFORMANCE: {metrics_str}")

class LoggingContextManager:
    """Context manager for logging function execution."""
    
    def __init__(self, func_name: str, logger: logging.Logger = None):
        self.func_name = func_name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.func_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.debug(f"Completed {self.func_name} in {execution_time:.3f}s")
            log_performance(self.func_name, execution_time)
        else:
            self.logger.error(f"Error in {self.func_name} after {execution_time:.3f}s: {exc_val}")
        
        return False  # Don't suppress exceptions

def performance_monitor(func):
    """
    Decorator to monitor function performance.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        logger = get_logger()
        
        with LoggingContextManager(func_name, logger):
            return func(*args, **kwargs)
    
    return wrapper
