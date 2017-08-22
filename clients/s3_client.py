import boto3
import constants as c
from .client_constants import *


# List Files

def list_experiment_configs():
    return list_fnames(EXPERIMENT_CONFIG_PREFIX,
                       c.EXPERIMENT_CONFIG_FILE_EXT)

def list_experiments():
    return list_fnames(EXPERIMENT_PREFIX, c.EXP_FILE_EXT)

def list_predictions():
    return list_fnames(PREDICTION_PREFIX, c.PRED_FILE_EXT)

def list_fnames(prefix, postfix):
    keys = get_keys(prefix=prefix)
    names = []
    for k in keys:
        names.append(k.replace(prefix,'').replace(postfix,''))
    return names


# Download

def download_experiment(dest_fpath, exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_PREFIX+exp_name + c.EXP_FILE_EXT
    download_file(dest_fpath, key, bucket=bucket)

def download_experiment_config(dest_fpath, exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_CONFIG_PREFIX+exp_name+c.EXPERIMENT_CONFIG_FILE_EXT
    download_file(dest_fpath, key, bucket=bucket)

def download_prediction(dest_fpath, pred_name, bucket=S3_BUCKET):
    key = PREDICTION_PREFIX+pred_name+c.PRED_FILE_EXT
    download_file(dest_fpath, key, bucket=bucket)


# Read Object directly from S3

def fetch_experiment_history(exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_HISTORY_PREFIX+exp_name+c.EXPERIMENT_HISTORY_FILE_EXT
    return get_object_str(key, bucket)

def fetch_experiment_config(exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_CONFIG_PREFIX+exp_name+c.EXPERIMENT_CONFIG_FILE_EXT
    return get_object_str(key, bucket)


# Upload

def upload_experiment(src_fpath, exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_PREFIX+exp_name+c.EXP_FILE_EXT
    upload_file(src_fpath, key, bucket=bucket)

def upload_experiment_config(src_fpath, exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_CONFIG_PREFIX+exp_name+c.EXPERIMENT_CONFIG_FILE_EXT
    upload_file(src_fpath, key, bucket=bucket)

def upload_experiment_history(src_fpath, exp_name, bucket=S3_BUCKET):
    key = EXPERIMENT_HISTORY_PREFIX+exp_name+c.EXPERIMENT_HISTORY_FILE_EXT
    upload_file(src_fpath, key, bucket=bucket)

def upload_prediction(src_fpath, pred_name, bucket=S3_BUCKET):
    key = PREDICTION_PREFIX+pred_name+c.PRED_FILE_EXT
    upload_file(src_fpath, key, bucket=bucket)


# Cleanup

def delete_experiment(exp_name):
    exp_config_key = (EXPERIMENT_CONFIG_PREFIX + exp_name
                      + c.EXPERIMENT_CONFIG_FILE_EXT)
    exp_history_key = (EXPERIMENT_HISTORY_PREFIX + exp_name
                      + c.EXPERIMENT_HISTORY_FILE_EXT)
    delete_object(S3_BUCKET, key=exp_config_key)
    delete_object(S3_BUCKET, key=exp_history_key)


# Base Helpers

def get_client():
    return boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)

def get_resource():
    return boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY)

def get_buckets():
    return get_client().list_buckets()

def get_object_str(key, bucket=S3_BUCKET):
    s3 = get_resource()
    obj = s3.Object(bucket, key)
    return obj.get()['Body'].read().decode('utf-8')

def get_keys(prefix, bucket=S3_BUCKET):
    objs = get_objects(prefix, bucket)
    keys = []
    if 'Contents' not in objs:
        return keys
    for obj in objs['Contents']:
        keys.append(obj['Key'])
    return keys

def download_file(dest_fpath, s3_fpath, bucket=S3_BUCKET):
    get_client().download_file(Filename=dest_fpath,
                               Bucket=bucket,
                               Key=s3_fpath)

def upload_file(src_fpath, s3_fpath, bucket=S3_BUCKET):
    get_client().upload_file(Filename=src_fpath,
                             Bucket=bucket,
                             Key=s3_fpath)

def get_download_url(s3_path, bucket=S3_BUCKET, expiry=86400):
    return get_client().generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket,
                'Key': s3_path},
        ExpiresIn=expiry
    )

#key = 'experiment_configs/JeremyCNN-SGD-lr1-wd0001-bs32-id6E878.json'
def delete_object(bucket, key):
    return get_client().delete_object(
        Bucket=bucket,
        Key=key
    )
