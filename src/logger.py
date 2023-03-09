import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),'logs', LOG_FILE) # All logs will be created at mlproject/logs/<LOG_FILE>.

os.makedirs(logs_path, exist_ok=True) # Make the directory even if it exists.

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
print(LOG_FILE_PATH)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)s %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

