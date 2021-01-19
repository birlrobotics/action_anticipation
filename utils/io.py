import json
import os

def read(file_name, read_type='read', mode='r'):
    with open(file_name, mode) as f:
        if read_type == 'read':
            return f.read()
        elif read_type == 'readline':
            return f.readline()
        else:
            return f.readlines()

def write(file_name, data, mode="w"):
    with open(file_name, mode) as f:
        if isinstance(data, list):
            for i in data:
                f.write(str(i) + ' \n')
        else:
            f.write(data)

def loads_json(file_name):
    return json.loads(read(file_name))

def dumps_json(object, file_name):
    data = json.dumps(object, sort_keys=True, indent=4)
    write(file_name, data, 'w')

def mkdir_if_not_exists(dir_name, recursive=False):
    if os.path.exists(dir_name):
        return
    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)