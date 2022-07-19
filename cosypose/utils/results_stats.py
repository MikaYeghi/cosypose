import json
import os

"""Functions"""
def read_file(file_path):
    """This function loads the json file into a dictionary."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

def get_files():
        """This function assumes that the script is inside the cosypose BOP eval results directory (i.e. with files, such as 'errors_xxxxxx.json')."""
        files = os.listdir()                            # select all the files
        files_ = [x for x in files if "errors" in x]    # select only the errors files
        return files_

def load_results(files_paths):
    """This function loads the cosypose evaluation results."""
    files_paths = get_files()
    files = list()
    for file in files_paths:
        result = read_file(file)
        files.append(result)
    return files

"""Main code"""
files_paths = get_files()
results = load_results(files_paths)