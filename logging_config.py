import logging
import sys

def setup_logging():
    # Get the root logger
    root_logger = logging.getLogger('')

    # Set the root logger level to the lowest desired level.
    # This allows all messages to be passed to handlers, which can have their own levels.
    root_logger.setLevel(logging.DEBUG)

    # Make the setup idempotent. If handlers already exist, do nothing.
    # This prevents duplicate handlers if the module is imported multiple times.
    if root_logger.handlers:
        return

    # --- Create File Handler ---
    # This handler writes log messages of level INFO and above to a file.
    file_handler = logging.FileHandler('headline_analysis.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # --- Create Console Handler ---
    # This handler writes log messages of level DEBUG and above to the console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger('urllib3').setLevel(logging.WARNING) # Optional: Quiets noisy libraries
    logging.getLogger('newspaper').setLevel(logging.WARNING) # Optional: Quiets noisy libraries


# Run the setup function when this module is imported
setup_logging()