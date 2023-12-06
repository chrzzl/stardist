import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from IPython.display import Markdown
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import label
from tqdm import tqdm

from stardist import fill_label_holes, calculate_extents, Rays_GoldenSpiral, star_dist3D

'''This class mainly holds information about all instances of a given list of instance segmentation masks of worms.'''
class InstanceCollection:
    def __init__(self, instance_masks, instance_details):
        self.instance_masks = instance_masks
        self.instance_details = instance_details # TODO: Process instance_details if using Stardist predictions

        self.num_worms = len(instance_masks)
        # TODO: Multiprocessing
        # self.worms = self.create_worms(num_jobs)
        # def create_worms(self, num_jobs):
        #     return Parallel(n_jobs=num_jobs)(
        #         delayed(WormSegmentation)(worm_id, self.instance_masks[worm_id]) for worm_id in range(self.num_worms)
        #     )
        self.worms = [WormSegmentation(worm_id, instance_masks[worm_id]) for worm_id in range(self.num_worms)]

        self.all_instances ={}
        for worm_idx in range(self.num_worms):
            self.all_instances.update(self.worms[worm_idx].instances)

    def plot_instance_slice(self, worm_id, instance_id, z_slice):
        mask = self.instance_masks[worm_id]
        instance = self.all_instances[(worm_id, instance_id)]
        box_min = instance.bounding_min
        box_max = instance.bounding_max
        box_limits = (box_min[1], box_max[1], box_min[0], box_max[0])
        print(mask.shape)

        # Determine the maximum value in the segmentation mask
        max_value = np.max(mask[:,:,z_slice])
        # Create a grayscale colormap with the maximum value
        colors = plt.cm.gray(np.linspace(0, 1, max_value + 1))
        # Set the color for a specific value (e.g., 99) to green
        colors[instance_id] = [0, 1, 0, 1]  # Green color
        # Create a custom colormap
        custom_colormap = ListedColormap(colors)

        # Plot the segmentation mask
        plt.figure(figsize=(16,10))
        plt.imshow(mask[:,:,z_slice], cmap=custom_colormap, vmin=0, vmax=max_value)
        plt.axis('off')
        plt.title(f'Mask slice z={z_slice} of instance {(worm_id, instance_id)}')
        plt.xlim(box_limits[0], box_limits[1])
        plt.ylim(box_limits[2], box_limits[3])
        plt.show()

    # A function that plots a GIF of all instance slices
    def plot_instance(self, worm_id, instance_id, save_gif = False):
        mask = self.instance_masks[worm_id]
        instance = self.all_instances[(worm_id, instance_id)]
        box_min = instance.bounding_min
        box_max = instance.bounding_max
        padding = 2
        mask_box = mask[box_min[0]-padding:box_max[0]+padding, box_min[1]-padding:box_max[1]+padding, box_min[2]-padding:box_max[2]+padding]

        # Determine the maximum value in the segmentation mask
        max_value = np.max(mask_box)
        # Create a grayscale colormap with the maximum value
        colors = plt.cm.gray(np.linspace(0, 1, max_value + 1))
        # Set the color for a specific value (e.g., 99) to green
        colors[instance_id] = [0, 1, 0, 1]  # Green color
        # Create a custom colormap
        custom_colormap = ListedColormap(colors)

        images = []
        # Create images from slices of the 3D array
        for z_slice in range(mask_box.shape[2]):
            # Apply the custom colormap to the slice
            slice_img = custom_colormap(mask_box[:, :, z_slice])
            # Convert the slice to an 8-bit unsigned integer array
            slice_img = (slice_img[:, :, :3] * 255).astype(np.uint8)  # Take only RGB channels
            slice_img = Image.fromarray(slice_img, 'RGB')
            images.append(slice_img)

        # Save the GIF
        random_id = np.random.randint(100000) # NOTE: This is a hack to avoid displaying a cached gif that will no get updated. - ck
        images[0].save(f'3d_array_slices_{random_id}.gif',
                    save_all=True, append_images=images[1:], optimize=False, duration=300, loop=0)
        if not save_gif:
            display(Markdown(f"Worm ID: {worm_id}, Instance ID: {instance_id}, Shape: {mask_box.shape}"))
            display(Markdown(f'<img src="3d_array_slices_{random_id}.gif" width="750" align="center">'))
            time.sleep(1)
            os.remove(f"3d_array_slices_{random_id}.gif")

    def clean_collection(self, problem_instances):
        # Remove problem instances from all_instances
        cleaned_instances = {instance_id: instance for instance_id, instance in self.all_instances.items() if instance_id not in problem_instances}
        # Copy object
        cleaned_collection = copy.deepcopy(self)
        cleaned_collection.all_instances = cleaned_instances
        return cleaned_collection
    

'''This class mainly holds information about all instances of a given worm instance segmentation.'''
class WormSegmentation:
    def __init__(self, id, mask):
        self.worm_id = id
        self.original_mask = mask
        self.mask = fill_label_holes(mask)
        self.worm_centroid = self.get_worm_centroid()

        self.instance_ids = self.get_instance_ids()
        self.instance_count = len(self.instance_ids)
        self.instances = {(self.worm_id, instance_id): Instance(instance_id, self.worm_id, self.worm_centroid, self.mask) for instance_id in self.instance_ids}


    def get_instance_ids(self):
        unique_values = np.unique(self.mask)
        return unique_values[unique_values != 0] # Remove background

    def get_worm_centroid(self):
        fg_pixels = np.where(self.mask > 0)
        return np.mean(fg_pixels, axis=1)

'''This class mainly holds information about a single instance, e.g. its centroid, volume and bounding box limits.'''
class Instance:
    def __init__(self, instance_id, worm_id, worm_centroid, mask):
        assert instance_id != 0, "Instance id cannot be 0 which is background"
        self.instance_id = instance_id
        self.worm_id = worm_id

        self.pixels = self.get_instance_pixels(mask)
        self.volume = len(self.pixels[0])
        self.absolute_centroid = self.get_instance_centroid() # w.r.t. to image origin
        self.relative_centroid = self.absolute_centroid - worm_centroid # w.r.t. to worm centroid
        self.bounding_min, self.bounding_max = self.get_bounding_limits()
        self.diameter = self.bounding_max - self.bounding_min # w.r.t to x,y and z axis

    def get_instance_pixels(self, mask):
        return np.where(mask == self.instance_id)

    def get_instance_centroid(self):
        return np.mean(self.pixels, axis=1)

    def get_bounding_limits(self):
        return np.min(self.pixels, axis=1), np.max(self.pixels, axis=1)
    



    

'''This class estracts and contains the features from an InstanceCollection that are important for metric learning.'''
class InstanceFeatures:
    def __init__(self, instance_collection):
        self.instance_collection = instance_collection

        self.centroids = {} # Relative centroids
        self.rays = None
        self.anisotropy = None
        self.distances = {}
        # NOTE: You can add more features here.
        self.representations = {}

    # Returns empirical anisotropy of labeled objects
    def calculate_anisotropy(self, mask): 
        extents = calculate_extents(mask)
        anisotropy = tuple(np.max(extents) / extents)
        return anisotropy

    def extract_features(self, num_rays, use_anisotropy):
        instance_masks = self.instance_collection.instance_masks
        # Create rays based on empirical anisotropy
        self.anisotropy = self.calculate_anisotropy(instance_masks)
        anisotropy = self.anisotropy if use_anisotropy else None
        self.rays = Rays_GoldenSpiral(num_rays, anisotropy)
        # Compute stardist array per instance mask
        stardist_arrays = []
        for mask in instance_masks:
            stardist_arrays.append(star_dist3D(mask, self.rays))
        for (worm_id, instance_id), instance in self.instance_collection.all_instances.items():
            centroid = np.round(instance.absolute_centroid).astype(int)
            ray_distances = stardist_arrays[worm_id][centroid[0], centroid[1], centroid[2], :]
            # Feature: Ray distances
            self.distances[(worm_id, instance_id)] = ray_distances
            # Feature: Relative centroids
            self.centroids[(worm_id, instance_id)] = instance.relative_centroid

            # Create instance representations as concatenated multidimensional array/vector
            self.representations[(worm_id, instance_id)] = np.concatenate((instance.relative_centroid, ray_distances), axis=0)
        print(f"Created instance representations (dim = {3+num_rays}) from:")
        print(f"- Relative centroids (dim = 3)")
        print(f"- Ray distances (dim = {num_rays})")

    def analyse_features(self, diameter_threshold):
        if not self.representations:
            print("No features to analyse. Call extract_features() first.")
            return
        
        problem_summary = {}
        # TODO: This loop is quite slow, make it more efficient.
        for (worm_id, instance_id), representation in tqdm(self.representations.items()): # For each instance:
            mask = self.instance_collection.instance_masks[worm_id]
            instance = self.instance_collection.all_instances[(worm_id, instance_id)]
            rounded_centroid = np.round(instance.absolute_centroid).astype(int)
            centroid_value = mask[tuple(rounded_centroid)]
            cc_count = count_connected_components(mask, instance_id)

            problems = []
            if 0 in set(representation[3:]): # If the distance from the centroid is 0:
                problems.append("Ray distance from centroid is 0.")
            if centroid_value == 0: # If centroid pixel is background:
                problems.append(f"Centroid pixel {rounded_centroid} is background.")
            if cc_count > 1: # If there are more than 1 connected components:
                problems.append(f"Number of connected components: {cc_count}")
            if any(d > threshold for d, threshold in zip(instance.diameter, diameter_threshold)): # If segmentation of instance is too large:
                problems.append(f"Too large instace diameter: {instance.diameter}")

            if problems:
                problem_summary[(worm_id, instance_id)] = problems
            print_problem_summary(problem_summary)
        return problem_summary


############ DATA CLEANING FUNCTIONS ############

def count_connected_components(mask, target_value):
    # Create array that has ones where mask equals target value
    array = mask == target_value
    structure = np.ones((3, 3, 3), dtype=int)
    _, ncomponents = label(array, structure)
    return ncomponents


def print_problem_summary(summary, verbose = False):
    if not summary:
        print("No problems found in instance collection. You can carry on! ^.^")
        return
    log_filename = "problem_log.txt"
    with open(log_filename, "w") as log_file:
        for instance, problems in summary.items():
            if problems:
                summary_strings = [f"\n- {problem}" for problem in problems]
                summary_string = "".join(summary_strings)
                if verbose:
                    print(f"{instance}: {summary_string}")
                log_file.write(f"{instance}: {summary_string}\n")
    
    print(f"Number of problematic instances: {len(summary)}\nSummary has been saved to {log_filename}.")
  