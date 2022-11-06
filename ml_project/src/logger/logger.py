import logging

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default_formatter': {
            'format': '[%(levelname)s:%(asctime)s] %(message)s'
        },
    },

    'handlers': {
        'file_handler': {
            'class': 'logging.FileHandler',
            'formatter': 'default_formatter',
            'filename': 'log.log'
        },
        'stream_handler': {
            'level': 'INFO',
            'formatter': 'default_formatter',
            'class': 'logging.StreamHandler',
        },
    },

    'loggers': {
        'logger': {
            'handlers': ['stream_handler', 'file_handler'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('logger')
