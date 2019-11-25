import os
import json

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)
files = [f for f in iter_files('/store/dyfar/wikipedia')]
for f in files:
    with open(f, encoding='utf-8') as fi:
        for line in fi:
            doc = json.loads(line)
            print(doc['text'].strip('\n\n\n').split('\n\n'))
            print('-------------------------------------------------')
