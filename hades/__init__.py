import os
import logging


def _configure_logger(logger):
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    return logger


def _apply_loglevel_from_env(logger, env='HADES_LOG_LEVEL'):
    name = os.getenv(env) or os.getenv(env.upper()) or os.getenv(env.lower())
    if not name:
        return
    levels = {'debug': logging.DEBUG, 'info': logging.INFO,
              'warning': logging.WARNING, 'error': logging.ERROR}
    requested_level = levels.get(name.lower())
    if requested_level:
        logger.setLevel(requested_level)


logger = logging.getLogger(__name__)
_configure_logger(logger)
_apply_loglevel_from_env(logger)
