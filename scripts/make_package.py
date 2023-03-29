from distutils.dir_util import copy_tree
import tarfile
import os


def merge(code_dir, model_dir):
    copy_tree(code_dir, model_dir)

def compress(tar_dir=None, output_file="model.tar.gz"):
    parent_dir = os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir('.'):
            print(item)
            tar.add(item, arcname=item)
    os.chdir(parent_dir)

def pack(code_dir, model_dir, output_file="model.tar.gz"):
    merge(code_dir, model_dir)
    compress(model_dir, output_file)

if __name__ == "__main__":
    code_dir = "code"
    model_dir = "model"
    output_file = "model.tar.gz"
    pack(code_dir, model_dir, output_file)
