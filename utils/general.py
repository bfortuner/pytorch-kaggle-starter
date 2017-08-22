import uuid


def gen_unique_id(prefix='', length=5):
    return prefix + str(uuid.uuid4()).upper().replace('-','')[:length]

def get_class_name(obj):
    invalid_class_names = ['function']
    classname = obj.__class__.__name__
    if classname is None or classname in invalid_class_names:
        classname = obj.__name__
    return classname

def dict_to_html(dd, level=0):
    """
    Convert dict to html using basic html tags
    """
    import simplejson
    text = ''
    for k, v in dd.items():
        text += '<br>' + '&nbsp;'*(4*level) + '<b>%s</b>: %s' % (k, dict_to_html(v, level+1) if isinstance(v, dict) else (simplejson.dumps(v) if isinstance(v, list) else v))
    return text

def dict_to_html_ul(dd, level=0):
    """
    Convert dict to html using ul/li tags
    """
    import simplejson
    text = '<ul>'
    for k, v in dd.items():
        text += '<li><b>%s</b>: %s</li>' % (k, dict_to_html_ul(v, level+1) if isinstance(v, dict) else (simplejson.dumps(v) if isinstance(v, list) else v))
    text += '</ul>'
    return text


