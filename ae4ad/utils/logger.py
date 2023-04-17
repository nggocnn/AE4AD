"""
Utility for logging
Reference: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
"""

import logging
from colorlog import ColoredFormatter


class AE4AD_Logger:
    logger = None

    @staticmethod
    def get_logger():
        if AE4AD_Logger.logger is None:
            LOG_LEVEL = logging.DEBUG
            LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
            logging.root.setLevel(LOG_LEVEL)
            formatter = ColoredFormatter(LOG_FORMAT)
            stream = logging.StreamHandler()
            stream.setLevel(LOG_LEVEL)
            stream.setFormatter(formatter)

            AE4AD_Logger.logger = logging.getLogger('pythonConfig')
            AE4AD_Logger.logger.setLevel(LOG_LEVEL)
            AE4AD_Logger.logger.addHandler(stream)

            AE4AD_Logger.logger.info('Logger initialized!')

            # AE4AD_Logger.logger.debug("A quirky message only developers care about")
            # AE4AD_Logger.logger.info("Curious users might want to know this")
            # AE4AD_Logger.logger.warning("Something is wrong and any user should be informed")
            # AE4AD_Logger.logger.error("Serious stuff, this is red for a reason")
            # AE4AD_Logger.logger.critical("OH NO everything is on fire")

        return AE4AD_Logger.logger
