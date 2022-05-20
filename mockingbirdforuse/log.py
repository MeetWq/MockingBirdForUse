import sys
import loguru

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from loguru import Logger

logger: "Logger" = loguru.logger


class Filter:
    def __init__(self) -> None:
        self.level: Union[int, str] = "DEBUG"

    def __call__(self, record):
        module_name: str = record["name"]
        record["name"] = module_name.split(".")[0]
        levelno = (
            logger.level(self.level).no if isinstance(self.level, str) else self.level
        )
        return record["level"].no >= levelno


logger.remove()
default_filter: Filter = Filter()
default_format: str = (
    "<g>{time:MM-DD HH:mm:ss}</g> "
    "[<lvl>{level}</lvl>] "
    "<c><u>{name}</u></c> | "
    "{message}"
)
logger.add(
    sys.stdout,
    level=0,
    colorize=True,
    diagnose=False,
    filter=default_filter,
    format=default_format,
)
