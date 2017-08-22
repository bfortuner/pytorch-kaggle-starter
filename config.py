import os
import socket
import init_project
import constants as c

# Main config
HOSTNAME = socket.gethostname()
PROJECT_NAME = 'dogscats'
PROJECT_PATH = '/bigguy/data/' + PROJECT_NAME
PROJECT_TYPE = c.SEGMENTATION
IMG_INPUT_FORMATS = [c.JPG]
IMG_TARGET_FORMATS = [c.BCOLZ] #segmentation or generative
IMG_DATASET_TYPES = [c.TRAIN, c.TEST]
METADATA_PATH = os.path.join(PROJECT_PATH, 'metadata.csv')
PATHS = init_project.init_paths(PROJECT_PATH, IMG_DATASET_TYPES,
    IMG_INPUT_FORMATS, IMG_TARGET_FORMATS)

# AWS Config
AWS_ACCESS_KEY = os.getenv('KAGGLE_AWS_ACCESS_KEY', 'dummy')
AWS_SECRET_KEY = os.getenv('KAGGLE_AWS_SECRET_ACCESS_KEY', 'dummy')
AWS_REGION = 'us-west-1'
AWS_SES_REGION = 'us-west-2'
ES_ENDPOINT = 'search-kagglecarvana-s7dnklyyz6sm2zald6umybeuau.us-west-1.es.amazonaws.com'
ES_PORT = 80
KIBANA_URL = 'https://search-kagglecarvana-s7dnklyyz6sm2zald6umybeuau.us-west-1.es.amazonaws.com/_plugin/kibana'
TIMEZONE = 'US/Pacific'

# External Resources
# If True, you must setup an S3 bucket, ES Instance, and SES address
S3_ENABLED = bool(os.getenv('KAGGLE_S3_ENABLED', False))
ES_ENABLED = bool(os.getenv('KAGGLE_ES_ENABLED', False))
EMAIL_ENABLED = bool(os.getenv('KAGGLE_SES_ENABLED', False))


# Email Notifications
ADMIN_EMAIL = 'bfortuner@gmail.com'
USER_EMAIL = 'bfortuner@gmail.com'
