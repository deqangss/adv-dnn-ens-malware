import os
import fileinput
import shutil

import hashlib
import random
import string
import base64
import re

try:
    basestring
except Exception:
    basestring = str

ENC_KEY = 'cab228a122d3486bac7fab148e8b5aba'

def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    get file paths given the directory
    :param base_dir: basic directory
    :param dir_ext: directory append at the rear of base_dir
    :param file_ext: file extension
    :return: set of file paths. Avoid the repetition
    """
    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif '.' not in file_ext:
                    _ext = '.' + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path,file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = ['']
    file_path_set = set()
    for ext in file_exts:
        file_path_set = file_path_set | set(get_file_name(os.path.join(base_dir, dir_ext), ext))

    return file_path_set

def retrive_all_dirs(base_dir, dir_ext):
    dir_ = os.path.join(base_dir,dir_ext)
    return set([os.path.join(dir_, sub_[0]) for sub_ in os.walk(dir_)])

def crypt_identifier(idf, seed=2345):
    if idf == '':
        return ''
    random.seed(seed)
    def md5_transform(idf):
        if isinstance(idf, basestring):
            return hashlib.md5(idf.encode('utf-8'))
        else:
            return hashlib.md5(idf)
    start_idx = random.choice(range(0, 8))
    length = random.choice(range(8, 16-start_idx))

    head_letter = random.choice(string.ascii_lowercase)
    return head_letter + md5_transform(idf).hexdigest()[start_idx:start_idx+length]

def random_string(code):
    def sha1_transform(code):
        if isinstance(code, basestring):
            return hashlib.sha1(code.encode('utf-8'))
        else:
            return hashlib.sha1(code)
    return random.choice(string.ascii_uppercase) + sha1_transform(code).hexdigest()[:8]

def string_on_code(code):
    def md5_transform(code):
        if isinstance(code, basestring):
            return hashlib.md5(code.encode('utf-8'))
        else:
            return hashlib.md5(code)
    return 'md5' + md5_transform(code).hexdigest()

def random_name(seed=2345, code = 'abc'):
    if not isinstance(seed, int):
        raise TypeError("Integer required.", type(seed), seed)
    random.seed(seed)
    sample_letters = [random.sample(string.ascii_letters, 1)[0] for _ in range(12)]
    return random.choice(string.ascii_uppercase) + random_string(code) + ''.join(sample_letters)

def apply_encryption(base_string):
    key = ENC_KEY * int(len(base_string) / len(ENC_KEY) + 1)
    xor_string = ''.join(chr(ord(x) ^ ord(y)) for (x, y) in zip(base_string, key))
    return base64.b64encode(xor_string.encode('utf-8')).decode('utf-8')

def read_file_by_fileinput(file_path, inplace = True):
    try:
        return fileinput.input(file_path, inplace=inplace)
    except IOError as ex:
        raise IOError(str(ex))

def read_full_file(file_path):
    try:
        with open(file_path, 'r') as rh:
            return rh.read()
    except IOError as ex:
        raise IOError("Cannot load file '{}'".format(file_path))

def write_whole_file(obj, file_path):
    try:
        with open(file_path, 'w') as wh:
            wh.write(obj)
    except IOError as ex:
        raise IOError("Cannot write file '{}'".format(file_path))

def load_json(json_path):
    try:
        import yaml
        with open(json_path, 'r') as rh:
            return yaml.safe_load(rh)
    except IOError as ex:
        raise IOError(str(ex) + ": Unable to load json file.")

def dump_json(obj_dict, file_path):
    try:
        import json
        with open(file_path, 'w+') as fh:
            json.dump(obj_dict, fh)
    except IOError as ex:
        raise IOError(str(ex) + ": Fail to dump dict using json toolbox")

def fix_invalid_id(comp_name, spec_chr = '@&'):
    comp_name = comp_name.replace('$;', spec_chr+';')
    comp_name = comp_name.replace('$/', spec_chr+'/')
    if comp_name[-1] == '$':
        comp_name = comp_name[:-1] + spec_chr
    while comp_name.find('$'+spec_chr) != -1:
        comp_name = comp_name.replace('$'+spec_chr, spec_chr+spec_chr)
    return comp_name


def defix_invalid_id(comp_name,spec_chr = '@&'):
    return comp_name.replace(spec_chr, '$')

def path_split(path):
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    file_name, ext_name = os.path.splitext(base_name)
    return dir_name, file_name, ext_name

def rename_file(src,dst):
    try:
        os.rename(src,dst)
    except IOError as ex:
        raise IOError("Cannot rename the file '{}' to '{}'".format(src,dst))

def rename_dir(old, new):
    try:
        os.renames(old, new)
    except IOError as ex:
        raise IOError("Cannot rename the folder '{}' to '{}'".format(old, new))

def rename_tree_dir(old_name, new_name):
    old_folder_names = re.findall(r'[^\/]*?\/', (old_name + '/').replace('//', '/'))
    new_folder_names = re.findall(r'[^\/]*?\/', (new_name + '/').replace('//', '/'))

    assert len(old_folder_names) == len(new_folder_names)

    for i in range(len(old_folder_names)):
        src_dir = ''.join(new_folder_names[:i]) + old_folder_names[i]
        dst_dir = ''.join(new_folder_names[:i+1])
        if os.path.exists(src_dir) and \
            src_dir != dst_dir:
            rename_dir(src_dir,dst_dir)

def copy_file(src, dst_dir, dst_name_w_ext = None):
    if not os.path.isfile(src):
        raise Exception("No such file.")

    if os.path.isdir(dst_dir):
        dst = os.path.join(dst_dir, dst_name_w_ext)
    elif os.path.isfile(dst_dir):
        assert os.path.basename(dst_dir) == dst_name_w_ext
        dst = dst_dir
    else:
        raise IOError("Cannot copy file to destination '{}'".format(dst_dir))

    if not os.path.exists(dst_dir):
        os.makedirs(os.path.dirname(dst_dir))

    shutil.copy(src, dst)

def get_sha256file(file_path):
  fh = open(file_path, 'rb')
  sha256 = hashlib.sha256()
  while True:
      data = fh.read(8192)
      if not data:
          break
      sha256.update(data)
  return sha256.hexdigest()


if __name__ == "__main__":
    print(apply_encryption('sendTextMessage')) # special symbols