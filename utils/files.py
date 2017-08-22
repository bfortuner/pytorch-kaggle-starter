import os
import random
from glob import glob
import shutil
import gzip
import pickle
import json
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED
import re
import bcolz



def get_fnames_from_fpaths(fpaths):
    fnames = []
    for f in fpaths:
        if isinstance(f, tuple):
            f = f[0]
        fnames.append(os.path.basename(f))
    return fnames


def get_matching_files_in_dir(dirpath, regex):
    fpaths = glob(os.path.join(dirpath,'*.*'))
    match_objs, match_fpaths = [], []
    for i in range(len(fpaths)):
        match = re.search(regex, fpaths[i])
        if match is not None:
            match_objs.append(match)
            match_fpaths.append(fpaths[i])
    return match_objs, match_fpaths


def zipdir(basedir, archivename):
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                absfn = os.path.join(root, fn)
                zfn = absfn[len(basedir)+len(os.sep):] #XXX: relative path
                z.write(absfn, zfn)


def unzipdir(archive_path, dest_path, remove=True):
    ZipFile(archive_path).extractall(dest_path)
    if remove:
        os.remove(archive_path)


def save_json(fpath, dict_):
    with open(fpath, 'w') as f:
        json.dump(dict_, f, indent=4, ensure_ascii=False)


def load_json(fpath):
    with open(fpath, 'r') as f:
        json_ = json.load(f)
    return json_


def pickle_obj(obj, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickle_obj(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def get_fname_from_fpath(fpath):
    return os.path.basename(fpath)


def get_paths_to_files(root, file_ext=None, sort=True, strip_ext=False):
    filepaths = []
    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(root):
        filepaths.extend(os.path.join(dirpath, f) 
            for f in filenames if file_ext is None or f.endswith(file_ext))
        fnames.extend([f for f in filenames if file_ext is None or f.endswith(file_ext)])
    if strip_ext:
        fnames = [os.path.splitext(f)[0] for f in fnames]
    if sort:
        return sorted(filepaths), sorted(fnames)
    return filepaths, fnames


def get_random_image_path(dir_path):
    filepaths = get_paths_to_files(dir_path)[0]
    return filepaths[random.randrange(len(filepaths))]


def save_obj(obj, out_fpath):
    with open(out_fpath, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(fpath):
    return pickle.load(open(fpath, 'rb'))


def save_bcolz_array(fpath, arr):
    c=bcolz.carray(arr, rootdir=fpath, mode='w')
    c.flush()


def load_bcolz_array(fpath):
    return bcolz.open(fpath)[:]


def compress_file(fpath):
    gzip_fpath = fpath+'.gz'
    with open(fpath, 'rb') as f_in:
        with gzip.open(gzip_fpath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return gzip_fpath


def write_lines(fpath, lines, compress=False):
    lines_str = '\n'.join(lines)
    if compress:
        fpath += '.gz'
        lines_str = str.encode(lines_str)
        f = gzip.open(fpath, 'wb')
    else:
        f = open(fpath, 'w')
    f.write(lines_str)
    f.close()
    return fpath