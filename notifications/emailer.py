import config
from .email_constants import *
import clients.ses_client as ses
import utils.general


def send_experiment_status_email(exp, to_email):
    body = get_experiment_status_template(exp)
    ses.send_email(EXPERIMENT_STATUS_EMAIL['subject'], body, to_email)


def get_experiment_status_template(exp):
    status = exp.config.progress['status']
    msg = exp.config.progress['status_msg']
    progress = utils.general.dict_to_html_ul(exp.config.progress)
    config = exp.config.to_html()
    return EXPERIMENT_STATUS_EMAIL['body'] % (exp.name, status, msg,
                                              WEBSITE_URL, progress, config)
