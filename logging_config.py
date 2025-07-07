"""Centralized logging configuration for the clickbait analysis application.

This module sets up the root logger to ensure consistent logging across all
other modules in the project. It configures two handlers: one that writes
INFO-level (and higher) logs to a file, and another that writes DEBUG-level
(and higher) logs to the console. The setup is idempotent, preventing duplicate
handlers if the module is imported multiple times.
"""
import logging
import sys


def setup_logging():
    """Configures the root logger with file and console handlers."""
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

    # Create a file handler to write log messages to a file.
    # This handler is set to log messages of level INFO and higher.
    file_handler = logging.FileHandler('headline_analysis.log', mode='w')
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