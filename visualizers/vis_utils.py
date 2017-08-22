from . import kibana
from . import viz


VISUALIZERS = {
    'visdom': viz.load,
    'kibana': kibana.load
}

def get_visualizer(config, name):
    return VISUALIZERS[name.lower()](config)


def get_visualizers_from_config(config):
    visualizers = []
    for v in config.visualizers:
        visualizer = get_visualizer(config, v)
        visualizers.append(visualizer)
    return visualizers
