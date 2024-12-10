import numpy as np
import tqdm
from config import PARAMS
from utils import *

# Define the neighborhood for erosion simulation (dx, dy, distance)
NEIGHBORS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), 
             (-1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))]

class SandSimulator:
    def __init__(self, grid_size=PARAMS["grid_size"], random_seed=None):
        """
        Initialize the sand simulator with a height map and repose angle.
        :param grid_size: Tuple (rows, cols) specifying the size of the height map.
        :param random_seed: Integer to seed randomness for the initial terrain (None for flat terrain).
        """
        self.vehicule_trajectory = [] 
        self.grid_size = grid_size

        if random_seed is None:
            self.height_map = np.zeros(grid_size)  # Flat terrain
        else:
            self.height_map = generate_nonflat_field(grid_size, random_seed)  # Random terrain
            
        # maximum slope based on the repose angle
        self.maximum_slope = np.tan(np.radians(PARAMS["angle_of_respose"]))

    def compute_one_erosion_step(self):
        """
        Simulates a single erosion step on the height map and calculates the flux (q) between cells.
        """
        def avoid_vhl(i, j):
            """
            Checks if the given cell (i, j) is covered by the vehicule's mask.
            :param i: Row index.
            :param j: Column index.
            :return: True if the cell is covered by the vehicule's mask; False otherwise.
            """
            if self.vhl.position is None:
                return False
            mask_coord = tuple([i - self.vhl.position[0], j - self.vhl.position[1]])
            if (0 <= mask_coord[0] < self.vhl.mask.shape[0] and 
                0 <= mask_coord[1] < self.vhl.mask.shape[1] and 
                self.vhl.mask[mask_coord]):
                return True
            return False

        q = np.zeros_like(self.height_map)

        rows, cols = self.height_map.shape

        # Iterate through all cells except the borders
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
               
                # Skip cells occupied by the vehicule
                if avoid_vhl(i, j):
                    continue

                for di, dj, dist in NEIGHBORS:
                    ni, nj = i + di, j + dj

                    # Skip neighbor cells occupied by the vehicule
                    if avoid_vhl(ni, nj):
                        continue

                    slope = (self.height_map[ni, nj] - self.height_map[i, j]) / (PARAMS["cell_edge_length"] * dist)
                    if slope > self.maximum_slope:
                        flux = PARAMS["k"] * (slope - self.maximum_slope)
                        q[ni, nj] -= flux
                        q[i, j] += flux

        # Update the height map with the computed flux
        self.height_map += q
        return q
    
    def simulate_erosion(self):
        """
        Runs the erosion simulation until the flux is below a defined threshold.
        """
        with tqdm.tqdm(desc="Eroding", unit=" erosions", position=1,leave=False, dynamic_ncols=False) as pbar:
            q = self.compute_one_erosion_step()  # Perform the first erosion step
            pbar.update(1)
            
            # Continue until the total flux stabilizes below the threshold
            while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
                #HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))
                q = self.compute_one_erosion_step()
                pbar.update(1)
