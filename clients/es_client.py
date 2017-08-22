from elasticsearch import Elasticsearch
from datetime import datetime
import pytz

from .client_constants import *


def upload_experiment_history(config, history):
    index_docs(history.to_doc(config), ES_EXPERIMENT_HISTORY_INDEX,
                  ES_EXPERIMENT_HISTORY_DOC_TYPE)


def upload_experiment_config(config):
    index_doc(config.to_doc(), ES_EXPERIMENT_CONFIG_INDEX,
                 ES_EXPERIMENT_CONFIG_DOC_TYPE)


def upload_prediction(pred):
    index_doc(pred.to_doc(), ES_PREDICTIONS_INDEX,
                 ES_PREDICTIONS_DOC_TYPE)


def delete_experiment(config):
    delete_experiment_by_id(config.get_id())


def delete_experiment_by_id(exp_id):
    r1 = delete_by_field(ES_EXPERIMENT_HISTORY_INDEX,
                       ES_EXPERIMENT_HISTORY_DOC_TYPE,
                       ES_EXP_KEY_FIELD, exp_id)
    r2 = delete_by_field(ES_EXPERIMENT_CONFIG_INDEX,
                       ES_EXPERIMENT_CONFIG_DOC_TYPE,
                       ES_EXP_KEY_FIELD, exp_id)
    return r1,r2


def delete_experiment_by_field(field, value):
    r1 = delete_by_field(ES_EXPERIMENT_HISTORY_INDEX,
                   ES_EXPERIMENT_HISTORY_DOC_TYPE,
                   field, value)
    r2 = delete_by_field(ES_EXPERIMENT_CONFIG_INDEX,
                   ES_EXPERIMENT_CONFIG_DOC_TYPE,
                   field, value)
    return r1,r2


# API
# http://elasticsearch-py.readthedocs.io/en/master/api.html

def get_client():
    return Elasticsearch([
        {'host': ES_ENDPOINT, 'port': ES_PORT},
    ])

def create_index(name, shards=2, replicas=1):
    es = get_client()
    ok = es.indices.create(name, body={
        "settings" : {
            "index" : {
                "number_of_shards" : shards,
                "number_of_replicas" : replicas
            }
        }
    })['acknowledged']
    assert ok is True
    return ok

def delete_index(name):
    ok = get_client().indices.delete(name)['acknowledged']
    assert ok is True
    return ok

def delete_docs_by_ids(index_name, doc_ids):
    pass

def delete_by_field(index_name, doc_type, field, value):
    query = {
        "query": {
            "match" : {
                field : value
            }
        }
    }
    es = get_client()
    r = es.delete_by_query(index=index_name, doc_type=doc_type, body=query)
    return r

def search_by_field(index_name, doc_type, field, value):
    query = {
            "term" : {
                field : value
            }
    }
    print(query)
    es = get_client()
    resp = es.search(index=index_name, doc_type=doc_type, body=query)
    return resp

def get_doc(index_name, doc_key):
    return get_client().get(index_name, id=doc_key)

def search(index_name, query, metadata_only=False, n_docs=10):
    es = get_client()
    filters = []
    if metadata_only:
        filters = ['hits.hits._id', 'hits.hits._type']
    return es.search(index=index_name, filter_path=filters,
                     body=query, size=n_docs)

def index_doc(doc, index_name, doc_type):
    assert 'key' in doc
    es = get_client()
    doc['uploaded'] = datetime.now(pytz.timezone(TIMEZONE))
    es.index(index=index_name, doc_type=doc_type, body=doc, id=doc['key'])

def index_docs(docs, index_name, doc_type):
    # There exists a bulk API, but this is fine for now
    for doc in docs:
        index_doc(doc, index_name, doc_type)

def get_mappings(index_name):
    # Shows the keys and data types in an index
    return get_client().indices.get_mapping(index_name)

def doc_exists(index_name, doc_type, doc_id):
    return get_client().exists(index_name, doc_type, doc_id)

def health():
    return get_client().cluster.health(wait_for_status='yellow',
                                       request_timeout=1)

def ping():
    return get_client().ping()

