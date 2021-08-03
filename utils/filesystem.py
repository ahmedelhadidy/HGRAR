import os, shutil, glob
from os.path import expanduser


def delete_all_in(dir):
    if os.path.isfile(dir) or os.path.islink(dir):
        try:
            os.unlink(dir)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (dir, e))
    else:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_all_directories_contains(base_dire, pattern):
    if  not os.path.exists(base_dire) or  not os.path.isdir(base_dire):
        raise Exception(' %s is not exist or not directory' % base_dire)
    search_q = join(base_dire,'**',pattern)
    results = glob.glob(search_q, recursive=True)
    dirs = list([ os.path.split(res)[0] for res in results ])
    return dirs


def delete(*args):
    for filename in glob.glob(os.path.join(args[0],*args[1:],'**')):
        delete_all_in(filename)


def get_relative_to_home(*args):
    return os.path.join(expanduser("~"), *args)


def join(path, *paths):
    return os.path.join(path, *paths)


def create_path(path, exist_ok=False):
    os.makedirs(path, exist_ok=exist_ok)


def split(path):
    return os.path.split(path)

if __name__ == '__main__':
    get_all_directories_contains('/Users/ahmedelhadidy/test_model_retrain','saved_model.pb')