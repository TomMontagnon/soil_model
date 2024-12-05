import numpy as np
import tqdm
from config import PARAMS
from utils import *

NEIGHBORS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))]

class SandSimulator:
    def __init__(self, grid_size=PARAMS["grid_size"], random_seed=None):
        """
        Initialize the sand simulator with a height map and repose angle.
        :param grid_size: Tuple (rows, cols) for the height map size.
        :param random_seed: Integer for the random seed (if None, the soil will be flat)
        """
        self.object_trajectory = []  # Store the object's trajectory
        self.grid_size = grid_size
        
        if random_seed is None:
            self.height_map = np.zeros(grid_size)
        else:
            self.height_map = generate_nonflat_field(grid_size, random_seed)
            
        self.maximum_slope = np.tan(np.radians(PARAMS["angle_of_respose"]))  # Convert to maximum slope

    def compute_one_erosion_step(self):
        """
        Simulates one erosion step on the height map and calculates q (flux).
        """
        # Define the 8-neighbor coordinates (Y, X, distance)

        # Initialize the q array to store the flux between grid points
        q = np.zeros_like(self.height_map)

        # Get the dimensions of the height map
        rows, cols = self.height_map.shape

        # Iterate through each cell of the height map
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
               
                # Sand does not flow at the object coord
                mask_coord = tuple([i-self.obj.position[0], j-self.obj.position[1]])
                if 0 <= mask_coord[0] < self.obj.mask.shape[0] and \
                    0 <= mask_coord[1] < self.obj.mask.shape[1] and \
                    self.obj.mask[mask_coord]:
                    continue

                for di, dj, dist in NEIGHBORS:
                    ni, nj = i + di, j + dj
                    
                    # Sand does not flow on the object
                    mask_coord = tuple([ni-self.obj.position[0], nj-self.obj.position[1]])
                    if 0 <= mask_coord[0] < self.obj.mask.shape[0] and \
                        0 <= mask_coord[1] < self.obj.mask.shape[1] and \
                        self.obj.mask[mask_coord]:
                        continue

                    # Compute the slope between the current cell and the neighbor
                    slope = (self.height_map[ni, nj] - self.height_map[i, j]) / (PARAMS["cell_edge_length"]*dist) # DH / DX

                    # Calculate the flux based on the slope
                    if slope > self.maximum_slope:
                        flux = PARAMS["k"]  * (slope - self.maximum_slope)
                        q[ni, nj] -= flux
                        q[i, j] += flux
        self.height_map += q 
        return q
    
    def simulate_erosion(self):
        with tqdm.tqdm(desc="Eroding", unit=" erosions") as pbar:
            q = self.compute_one_erosion_step()
            pbar.update(1)
            while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
                q = self.compute_one_erosion_step()
                pbar.update(1)  
    
