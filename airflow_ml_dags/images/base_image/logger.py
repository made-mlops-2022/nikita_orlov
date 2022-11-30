LOGS_PATH = "/logs/total.log"
log_conf = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(levelname)s\t$(message)s"
        },
        "complex": {
            "format": "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s\t"
        }
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": LOGS_PATH,
            "formatter": "complex"
        },
        "stream_handler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple"
        }
    },
    "loggers": {
        "file_stream": {
            "level": "DEBUG",
            "handlers": ["file_handler", "stream_handler"]
        },
        "file": {
            "level": "DEBUG",
            "handlers": ["file_handler"]
        }
    }
}
