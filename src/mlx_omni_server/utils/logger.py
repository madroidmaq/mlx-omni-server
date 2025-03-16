import logging
import os
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Create logs directory
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get project logger with optimized Rich configuration

    Args:
        name: Optional module name for the logger

    Returns:
        logging.Logger: Configured logger instance with Rich handler
    """
    # Create console with no file/line highlighting
    console = Console(highlight=False)

    # Custom time formatter that only shows time (no date)
    def time_formatter():
        return Text(
            datetime.now().strftime("%H:%M:%S"), style="bold"
        )  # Only show hours:minutes:seconds

    # Configure Rich handler with custom settings
    rich_handler = RichHandler(
        console=console,
        show_time=False,  # Disable default time display
        show_level=True,
        show_path=False,  # Hide file path
        enable_link_path=False,  # Disable clickable links
        markup=True,
        rich_tracebacks=True,
        tracebacks_extra_lines=2,
        tracebacks_show_locals=True,
    )

    # Set custom time display function
    rich_handler.get_time = time_formatter

    # Set log format to only include the message
    # Rich handler will add timestamps and log levels automatically
    FORMAT = "%(message)s"

    # Configure the root logger
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        handlers=[rich_handler],
    )

    # Get the named logger or use 'mlx_omni' as default
    logger_name = name if name else "mlx_omni"
    log = logging.getLogger(logger_name)

    return log


# Default logger
logger = get_logger()
