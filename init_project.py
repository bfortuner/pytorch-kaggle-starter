import os
import config as cfg


def init_paths(root, dset_types, input_img_exts, target_img_exts):
    paths = {
        'project': root,
        'experiments': os.path.join(root, 'experiments'),
        'predictions': os.path.join(root, 'predictions'),
        'submissions': os.path.join(root, 'submissions'),
        'folds': os.path.join(root, 'folds')
    }
    for key in paths:
        os.makedirs(paths[key], exist_ok=True)
    
    paths['datasets'] = {}
    datasets_root = os.path.join(root, 'datasets')
    os.makedirs(datasets_root, exist_ok=True)
    make_dataset(paths, datasets_root, 'inputs', dset_types, input_img_exts)
    make_dataset(paths, datasets_root, 'targets', dset_types, target_img_exts)

    return paths


def make_dataset(paths, datasets_root, name, dset_types, img_exts):
    root = os.path.join(datasets_root, name)
    os.makedirs(root, exist_ok=True)
    paths['datasets'][name] = {}

    for dset in dset_types:
        for img in img_exts:
            dir_name = dset+'_'+img
            dir_path = os.path.join(root, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            paths['datasets'][name][dir_name] = dir_path


if __name__ == '__main__':
    init_paths(cfg.PROJECT_PATH, cfg.IMG_DATASET_TYPES, 
        cfg.IMG_INPUT_FORMATS, cfg.IMG_TARGET_FORMATS)