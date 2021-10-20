from logging import raiseExceptions
from multiprocessing import Value
import pickle
from matplotlib import pyplot as plt
import pyquaternion
import math
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from pprint import pprint
import pdb

class Dataloader:
    def __init__(self, file_path, objects_list=list(), xs=list(), ys=list()) -> None:
        self.file_path = file_path
        self.objects_list = objects_list
        self.xs = xs
        self.ys = ys
        self.authors_results = {
            1: 439,
            2: 307,
            3: 353,
            4: 27,
            5: 330,
            6: 170,
            7: 412,
            8: 300,
            9: 190,
            10: 439,
            11: 69,
            12: 103,
            13: 125,
            14: 294,
            15: 337,
            16: 327,
            17: 452,
            18: 108,
            19: 83,
            20: 58
        }
    
    def load_data(self, verbose=0):
        with open(self.file_path, 'rb') as inp:
            while True:
                try:
                    new_object = pickle.load(inp)
                    if None not in (new_object.coarse_error, new_object.refiner_error):
                        self.objects_list.append(new_object)
                        self.xs.append(new_object.coarse_error)
                        self.ys.append(new_object.refiner_error)
                        if verbose == 1:
                            print(new_object)
                except Exception:
                    print(f"Finished loading objects. Total number of objects: {len(self.objects_list)}")
                    break
    
    def update_data(self, new_data):
        self.objects_list = new_data
        xs, ys = list(), list()
        for obj in self.objects_list:
            xs.append(obj.coarse_error)
            ys.append(obj.refiner_error)
        self.xs = xs
        self.ys = ys

    def plot_data(self, objects_selected=None, separation_line=False):
        if objects_selected is None: # plot all object
            self.plot_custom_data(self.xs, self.ys, separation_line=separation_line)
        else:
            xs, ys = list(), list()
            for obj in self.objects_list:
                if obj.object_id in objects_selected:
                    xs.append(obj.coarse_error)
                    ys.append(obj.refiner_error)
            self.plot_custom_data(xs, ys, separation_line=separation_line)
    
    def plot_custom_data(self, xs, ys, separation_line=False):
        assert len(xs) == len(ys)
        # Plotting
        plt.plot(xs, ys, 'bo')
        v = min([max(xs), max(ys)])
        if separation_line:
            plt.plot([0, v], [0, v], 'r')
        plt.xlabel("Coarse error")
        plt.ylabel("Refiner error")

        # Calculating percentage of improved data points
        total_n_points = len(xs)
        improved_n_points = 0
        for x in range(total_n_points):
            if ys[x] < xs[x]:
                improved_n_points += 1
        percentage_improved = round(improved_n_points / total_n_points * 100, 2)
        print(f"Percentage of improved points: {percentage_improved}%")

        plt.show()
    
    def filter_objects(self, ref_min=0, ref_max=1e6, coarse_min=0, coarse_max=1e6):
        relevant_objects = list()
        for obj in self.objects_list:
            if obj.refiner_error >= ref_min and obj.refiner_error <= ref_max and obj.coarse_error >= coarse_min and obj.coarse_error <= coarse_max:
                relevant_objects.append(obj)
        return relevant_objects

    def get_data(self):
        return self.objects_list

    def get_object_by_scene_view_label(self, scene_id, view_id, object_id):
        obj = None
        for obj_ in self.objects_list:
            if obj_.scene_id == scene_id and obj_.view_id == view_id and obj_.object_id == object_id:
                obj = obj_
                break
        assert obj is not None
        return obj

    def get_objects_unique_count(self):
        objects_count = dict()
        for obj in tqdm(self.objects_list):
            if obj.object_id in list(objects_count.keys()): # if object has already been recorded at least once
                objects_count[obj.object_id] += 1
            else: # if object is encountered for the first time
                objects_count[obj.object_id] = 1
        return objects_count
    
    def get_objects_count(self):
        objects_count = self.get_objects_unique_count()
        count = 0
        for x in list(objects_count.keys()):
            count += objects_count[x]
        return count

    def get_authors_only(self):
        authors_objects = list()
        for obj in tqdm(self.objects_list):
            obj_scene_id = obj.scene_id
            obj_view_id = obj.view_id
            if obj_view_id <= self.authors_results[obj_scene_id]:
                authors_objects.append(obj)
        return authors_objects
    
    def plot_error_vs_angle(self, error_type):
        assert error_type in ('refiner', 'coarse'), "Error type must be either 'refiner' or 'coarse'"

        xs, ys = list(), list()
        for obj in self.objects_list:
            # Extract distance angle
            obj_distortion = obj.distortion
            transform = pyquaternion.Quaternion(axis=obj_distortion[:3], angle=obj_distortion[3]).transformation_matrix
            distance_angle = math.acos((transform.trace() - 2) / 2)
            xs.append(distance_angle)

            # Extract error
            if error_type == 'refiner':
                ys.append(obj.refiner_error)
            elif error_type == 'coarse':
                ys.append(obj.coarse_error)
            else:
                raise ValueError("Error type must be either 'coarse' or 'refiner'")
            
        plt.plot(xs, ys, 'bo')
        plt.xlabel("Distance angle, radians")
        plt.ylabel("Error value")
        plt.title("Error vs angle")
        plt.grid(True)
        plt.show()

# dl = Dataloader(file_path="/home/yemika/Mikael/Oxford/Studying/4YP/code/cosypose/local_data/results/tless-siso-n_views=1--4312481950/results_GPU_1.txt")
# dl = Dataloader(file_path="/home/yemika/Mikael/Oxford/Studying/4YP/code/cosypose/local_data/results/tless-siso-n_views=1--7937853015/results_GPU_0.txt")
dl = Dataloader(file_path="/home/yemika/Mikael/Oxford/Studying/4YP/code/cosypose/local_data/results/tless-siso-n_views=1--2572992757/multi_initializations_results.pkl")

# Loading the data
dl.load_data()
# dl.update_data(dl.get_authors_only())
print(dl.get_data())

# Filtering
# filtered = dl.filter_objects(ref_min=0.14)
# for x in dl.get_data():
#     print(x)

# Plotting the data
# dl.plot_data(separation_line=True)
# print(sum(list(dl.authors_results.values())))