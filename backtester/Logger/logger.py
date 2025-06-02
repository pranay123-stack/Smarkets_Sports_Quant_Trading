import logging
import os
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler

class Logger:
    def __init__(self, log_dir="Logs"):
        self.console = Console()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"log_{timestamp}.log")

        logging.basicConfig(
            level="NOTSET",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console), logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger("rich")

    def log(self, message, level="info"):
        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
