"""Centralized logging configuration for the clickbait analysis application.

This module sets up the root logger to ensure consistent logging across all
other modules in the project. It configures two handlers: one that writes
INFO-level (and higher) logs to a daily file, and another that writes
DEBUG-level (and higher) logs to the console. The setup is idempotent,
preventing duplicate handlers if the module is imported multiple times.
"""
import logging
import sys
from datetime import datetime


def setup_logging():
    """Configures the root logger with daily file and console handlers."""
    # Get the root logger, which is the ancestor of all other loggers.
    root_logger = logging.getLogger('')

    # Set the root logger's level to the lowest desired level. This allows
    # messages to be passed to handlers, which can have their own, more
    # restrictive levels.
    root_logger.setLevel(logging.DEBUG)

    # Make the setup idempotent. If handlers have already been added, this
    # function will do nothing, preventing duplicate log messages.
    if root_logger.handlers:
        return

    # --- Create a daily log file ---
    # Get the current date to create a unique filename for each day.
    current_date = datetime.now().strftime('%d_%m_%Y')
    log_filename = f'headline_analysis_{current_date}.log'

    # Create a file handler to write log messages to the daily file.
    # mode='a' ensures that if the file exists, logs are appended. If not, it's created.
    # This handler is set to log messages of level INFO and higher.
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create a console handler to write log messages to standard output.
    # This handler is set to log messages of level DEBUG and higher.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Reduce the logging verbosity of noisy third-party libraries.
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('newspaper').setLevel(logging.WARNING)


# Run the setup function immediately when this module is imported. This ensures
# that logging is configured as early as possible in the application's lifecycle.
setup_logging()