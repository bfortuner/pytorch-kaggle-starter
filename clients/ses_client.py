import boto3
from .client_constants import *


def get_client():
    return boto3.client('ses', aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name=AWS_SES_REGION)


def send_email(subject, body, to_email, from_email=ADMIN_EMAIL):
    response = get_client().send_email(
        Source=from_email,
        Destination={
            'ToAddresses': [
                to_email,
            ],
            'CcAddresses': [],
            'BccAddresses': []
        },
        Message={
            'Subject': {
                'Data': subject,
                'Charset': EMAIL_CHARSET
            },
            'Body': {
                'Text': {
                    'Data': body,
                    'Charset': EMAIL_CHARSET
                },
                'Html': {
                    'Data': body,
                    'Charset': EMAIL_CHARSET
                }
            }
        }
    )
    return response

