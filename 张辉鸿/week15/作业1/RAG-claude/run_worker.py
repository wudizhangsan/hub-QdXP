"""Start the Kafka-based document parsing worker."""

import logging
from app.worker.parse_worker import run_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    run_worker()
