import config
import constants as c


WEBSITE_URL = config.KIBANA_URL
ADMIN_EMAIL = config.ADMIN_EMAIL
USER_EMAIL = config.USER_EMAIL
EMAIL_CHARSET = 'UTF-8'

HEADER="<html>"
FOOTER="</html>"

EXPERIMENT_STATUS_EMAIL_TEMPLATE="""
<p>Hello,</p>
<p>Your experiment has ended.</p>
<p><b>Name:</b> %s</p>
<p><b>Status:</b> %s</p>
<p><b>Status Msg:</b> %s</p>
<p><a href="%s">View Dashboard</a></p>
<p><b>Experiment Results:</b></p>
<p>%s</p>
<p><b>Experiment Config:</b></p>
<p>%s</p>
<p><b>Thanks,<br>
Team</p>
"""

EXPERIMENT_STATUS_EMAIL_BODY = (
   HEADER + EXPERIMENT_STATUS_EMAIL_TEMPLATE + FOOTER
)

EXPERIMENT_STATUS_EMAIL ={
    'subject' : 'New Experiment Results',
    'body' : EXPERIMENT_STATUS_EMAIL_BODY
}
