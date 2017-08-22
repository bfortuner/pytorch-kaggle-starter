import os
import logging
import imp
import time


def get_logger(log_path='',
               logger_name='logger',
               ch_log_level=logging.ERROR,
               fh_log_level=logging.INFO):
    logging.shutdown()
    imp.reload(logging)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Console Handler
    if ch_log_level:
        ch = logging.StreamHandler()
        ch.setLevel(ch_log_level)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    # File Handler
    if fh_log_level:
        fh = logging.FileHandler(os.path.join(log_path,logger_name+'.log'))
        fh.setLevel(fh_log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_time_msg(start_time):
    time_elapsed = time.time() - start_time
    msg = 'Time {:.1f}m {:.2f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    return msg