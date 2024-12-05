import logging
import os

# ANSI escape codes for colors
COLORS = {
    "verbose": "\x1b[38;5;244m",  # Grey
    "debug": "\x1b[38;5;33m",  # Blue (#2196F3)
    "info": "\x1b[38;5;77m",  # Green
    "warning": "\x1b[38;5;220m",  # Yellow
    "error": "\x1b[38;5;196m",  # Red
    "critical": "\x1b[38;5;199m",  # Purple
    "reset": "\x1b[0m",
}

# Create logs directory
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class ColorFormatter(logging.Formatter):
    """Custom color formatter for logging"""

    format_str = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: COLORS["debug"] + format_str + COLORS["reset"],
        logging.INFO: COLORS["info"] + format_str + COLORS["reset"],
        logging.WARNING: COLORS["warning"] + format_str + COLORS["reset"],
        logging.ERROR: COLORS["error"] + format_str + COLORS["reset"],
        logging.CRITICAL: COLORS["critical"] + format_str + COLORS["reset"],
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str = "mlx_omni") -> logging.Logger:
    """Get project logger

    Args:
        name: Module name, typically pass in __name__

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    _logger = logging.getLogger(name)  # Use module name directly as logger name

    # Set log level
    _logger.setLevel(os.environ.get("MLX_OMNI_LOG_LEVEL", "INFO").upper())

    # If logger already has handlers, it's already configured
    if _logger.handlers:
        return _logger

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    _logger.addHandler(console_handler)

    # Disable log propagation to parent logger
    _logger.propagate = False

    return _logger


# Default logger
logger = get_logger()
