"""Build Agents with tool use."""

from loguru import logger
from loguru._defaults import env
import sys

# https://github.com/Delgan/loguru/issues/51


logger.remove(0)
LOGURU_LEVEL = env("LOGURU_LEVEL", str, "INFO")
logger.start(sys.stderr, level=LOGURU_LEVEL)

__version__ = "0.0.1"
