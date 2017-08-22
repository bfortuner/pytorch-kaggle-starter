import config as cfg

# AWS Config
AWS_REGION = cfg.AWS_REGION
AWS_ACCESS_KEY = cfg.AWS_ACCESS_KEY
AWS_SECRET_KEY = cfg.AWS_SECRET_KEY
TIMEZONE = cfg.TIMEZONE

# S3 Config
S3_BUCKET = 'kaggle{:s}'.format(cfg.PROJECT_NAME)
EXPERIMENT_CONFIG_PREFIX = 'experiment_configs/'
EXPERIMENT_HISTORY_PREFIX = 'experiment_histories/'
EXPERIMENT_PREFIX = 'experiments/'
PREDICTION_PREFIX = 'predictions/'
ENSEMBLE_PREFIX = 'ensembles/'

# Elasticsearch Config
ES_EXPERIMENT_HISTORY_INDEX = 'kaggle-{:s}-history'.format(cfg.PROJECT_NAME)
ES_EXPERIMENT_CONFIG_INDEX = 'kaggle-{:s}-config'.format(cfg.PROJECT_NAME)
ES_PREDICTIONS_INDEX = 'kaggle-{:s}-predictions'.format(cfg.PROJECT_NAME)
ES_EXPERIMENT_HISTORY_DOC_TYPE = 'history'
ES_EXPERIMENT_CONFIG_DOC_TYPE = 'config'
ES_PREDICTIONS_DOC_TYPE = 'prediction'
ES_ENDPOINT = cfg.ES_ENDPOINT
ES_PORT = cfg.ES_PORT

# SES Config
AWS_SES_REGION = cfg.AWS_SES_REGION
ADMIN_EMAIL = cfg.ADMIN_EMAIL
USER_EMAIL = cfg.USER_EMAIL
EMAIL_CHARSET = 'UTF-8'
