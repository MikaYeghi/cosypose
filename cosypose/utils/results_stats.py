import json
import os
from matplotlib import pyplot as plt
import numpy as np

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

def generate_object_stats(results):
    """This function generates statistics for the 30 objects from the T-LESS data set using the results supplied."""
    stats = {i: list() for i in range(1, 31)}
    for scene in results:
        for obj in scene:
            error_val = max(obj['errors'].values())[0]
            obj_id = obj['obj_id']
            stats[obj_id].append(error_val)
    return stats
    
def plot_results(stats, idx=5, num_bins=10):
    """Variable idx represents either the object id or the scene id."""
    stats_ = stats[idx]
    
    # Build the figure and the histogram
    plt.hist(stats_, bins=np.linspace(0.0, 1.0, num=num_bins+1))
    plt.title("Errors histogram")
    plt.xlabel("Error value")
    plt.ylabel("Frequency")
    plt.show()


"""Main code"""
# Load the results
files_paths = get_files()
results = load_results(files_paths)

# Generate statistics per object
stats = generate_object_stats(results)
plot_results(stats, idx=5, num_bins=10)