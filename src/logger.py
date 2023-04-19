import logging
import os
import sys

# from src.exception import CustomException
from datetime import datetime

logs_path=os.path.join(os.getcwd(),"logs")

os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,"logging_details.log")

logging.basicConfig(
                    filename=LOG_FILE_PATH,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO,
                   )