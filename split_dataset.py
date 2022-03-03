import pdb
import os
import json
from tqdm import tqdm

# Functions
def update_data(scene_gt_info, scene_gt, labels_to_remove):
    updated_scene_gt = dict()
    updated_scene_gt_info = dict()
    samples_to_remove = dict() # view_id: [list of id-s to be removed]

    # Initialize empty arrays for each view
    for view_id in scene_gt.keys():
        samples_to_remove[view_id] = list()
    
    # Run through scene_gt to find labels to be removed
    for view_id in scene_gt.keys():
        for idx in range(len(scene_gt[view_id])):
            if scene_gt[view_id][idx]['obj_id'] in labels_to_remove:
                samples_to_remove[view_id].append(idx)

    # Use samples_to_remove to remove extra samples
    for view_id in scene_gt.keys():
        updated_scene_gt[view_id] = list()
        updated_scene_gt_info[view_id] = list()
        for idx in range(len(scene_gt[view_id])):
            if idx not in samples_to_remove[view_id]:
                updated_scene_gt[view_id].append(scene_gt[view_id][idx])
        for idx in range(len(scene_gt_info[view_id])):
            if idx not in samples_to_remove[view_id]:
                updated_scene_gt_info[view_id].append(scene_gt_info[view_id][idx])
        
        # Check if the view is empty
        if len(updated_scene_gt[view_id]) == 0:
            del updated_scene_gt[view_id]
        if len(updated_scene_gt_info[view_id]) == 0:
            del updated_scene_gt_info[view_id]

    return updated_scene_gt_info, updated_scene_gt

# Variables
labels_to_remove = [25, 26, 27, 28, 29, 30]

"""
Get the list of directories of each scene.
"""
dirs_list = os.listdir()
for idx in range(len(dirs_list)):
    if dirs_list[idx] == 'sort.py':
        del dirs_list[idx]

# Loop through all directories
for scene in tqdm(dirs_list):
    os.chdir(scene)
    
    with open('scene_gt_info.json') as json_file:
        scene_gt_info = json.load(json_file)
    with open('scene_gt.json') as json_file:
        scene_gt = json.load(json_file)
    
    updated_scene_gt_info, updated_scene_gt = update_data(scene_gt_info, scene_gt, labels_to_remove)

    # Check if there are empty views
    for view_id in updated_scene_gt.keys():
        if len(scene_gt[view_id]) == 0:
            print(f"Empty view detected! Scene: {scene}, view id: {view_id}")

    # Write down the updated dataset data
    with open('scene_gt.json', 'w', encoding='utf-8') as f:
        json.dump(updated_scene_gt, f, ensure_ascii=False, indent=4)
    with open('scene_gt_info.json', 'w', encoding='utf-8') as f:
        json.dump(updated_scene_gt_info, f, ensure_ascii=False)

    os.chdir('..')