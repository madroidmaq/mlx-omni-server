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
    # Attach the Rich handler to the root logger via basicConfig. Use NOTSET
    # so we can control the effective level later (via set_logger_level).
    # Use the integer constant instead of a string to avoid accidental misuse.
    logging.basicConfig(
        level=logging.NOTSET,
        format=FORMAT,
        handlers=[rich_handler],
    )

    # Get the named logger or use 'mlx_omni' as default
    logger_name = name if name else "mlx_omni"
    log = logging.getLogger(logger_name)

    return log


def set_logger_level(logger: logging.Logger, level: str):
    """Set logging level for the given logger and ensure it applies globally.

    We set the level on the provided logger, on the root logger, and on all
    existing handlers. This ensures messages from other modules (uvicorn,
    FastAPI, etc.) respect the requested level.
    """
    # Resolve textual level to numeric value, default to INFO if unknown
    log_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    if level.upper() not in logging.getLevelNamesMapping():
        logger.warning(f"Invalid log level '{level}', defaulting to INFO")

    # Set level on the provided logger
    logger.setLevel(log_level)

    # Also set the root logger level so that other loggers inherit it when
    # they have NOTSET level themselves.
    logging.root.setLevel(log_level)

    # Ensure all existing handlers respect the new level (RichHandler, etc.)
    for handler in logging.root.handlers:
        try:
            handler.setLevel(log_level)
        except (AttributeError, TypeError):
            # If a handler doesn't support setLevel or is of an unexpected
            # type, skip it.
            pass


# Default logger
logger = get_logger()
