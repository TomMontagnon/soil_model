import numpy as np
import tqdm
from config import *
from utils import *
from numba import jit

# Define the neighborhood for erosion simulation (dx, dy, distance)
NEIGHBORS = np.array([(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1,np.sqrt(2)), (1, 1,np.sqrt(2)), (-1, 1,np.sqrt(2)), (1, -1,np.sqrt(2))], dtype=np.float64)

class SandSimulator:
    def __init__(self, grid_size, random_seed=None):
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

    def simulate_erosion(self):
        """
        Runs the erosion simulation until the flux is below a defined threshold.
        """
        print(PARAMS["cell_edge_length"])
        with tqdm.tqdm(desc="Eroding", unit=" erosions", position=1,leave=False, dynamic_ncols=False) as pbar:
            # Perform the first erosion step
            q = compute_one_erosion_step(np.float64(PARAMS["k"]), self.height_map, 
                                         self.maximum_slope, self.vhl.position, self.vhl.mask)  
            pbar.update(1)
            
            # Continue until the total flux stabilizes below the threshold
            while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
                #HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))   
                #print(np.sum(np.abs(q)))
                q = compute_one_erosion_step(np.float64(PARAMS["k"]), self.height_map, 
                                             self.maximum_slope, self.vhl.position, self.vhl.mask)
                pbar.update(1)




@jit(nopython=True, parallel=True)
def compute_one_erosion_step(K, height_map, maximum_slope, vhl_pos, vhl_mask):
    print(K, maximum_slope, vhl_pos, vhl_mask.shape)
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
        if vhl_pos is None:
            return False
        mask_y = int(i - vhl_pos[0])
        mask_x = int(j - vhl_pos[1])
        if (0 <= mask_y < vhl_mask.shape[0] and  \
            0 <= mask_x < vhl_mask.shape[1] and  \
            vhl_mask[mask_y,mask_x]):
            return True
        return False

    q = np.zeros_like(height_map, dtype=np.float64)
    rows, cols = height_map.shape

    # Iterate through all cells except the borders
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
           
            # Skip cells occupied by the vehicule
            if avoid_vhl(i, j):
                continue

            for di, dj, dist in NEIGHBORS:
                ni, nj = int(i + di), int(j + dj)

                # Skip neighbor cells occupied by the vehicule
                if avoid_vhl(ni, nj):
                    continue
                slope = (height_map[ni, nj] - height_map[i, j]) / dist
                if slope > maximum_slope:
                    flux = (slope - maximum_slope)
                    q[ni, nj] -= flux
                    q[i, j] += flux

    # Update the height map with the computed flux
    height_map += K * q
    return q

