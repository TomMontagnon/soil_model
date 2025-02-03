import numpy as np
import tqdm
from config import *
from utils import *
from numba import jit
from numba import cuda, float64
from os import system

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
        self.vhl = None

        if random_seed is None:
            self.height_map = np.zeros(grid_size)  # Flat terrain
        else:
            self.height_map = generate_nonflat_field(grid_size, random_seed)  # Random terrain
            
        # maximum slope based on the repose angle
        self.maximum_slope = np.tan(np.radians(PARAMS["angle_of_respose"]))
        
        HM_STATES.append((self.height_map.copy(), np.array(self.vehicule_trajectory)))

    def final_erosion(self):
        self.vhl.position = None 
        self.simulate_erosion_cuda()
        HM_STATES.append((self.height_map.copy(), np.array(self.vehicule_trajectory)))


    def simulate_erosion_cuda(self, register=False):
        # Transfer data to GPU
        d_height_map = cuda.to_device(self.height_map)
        d_flux = cuda.to_device(np.zeros_like(self.height_map, dtype=np.float64))
        d_vhl_mask = cuda.to_device(self.vhl.mask)
    
        # Vehicle position and constants
        vhl_pos = self.vhl.position if self.vhl.position is not None else (-1, -1)
        d_vhl_pos = cuda.to_device(np.array(vhl_pos, dtype=np.int32))
    
        # Define thread and block configuration
        threads_per_block = (32, 32)
        blocks_per_grid_x = (self.height_map.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.height_map.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        # Initialize a progress bar to track the erosion process
        with tqdm.tqdm(desc="Eroding", unit=" erosions", position=1, leave=False, dynamic_ncols=False) as pbar:
            q = np.inf  # Initialize with a large value
            while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
            #for i in range(500):
                
                if register:
                    self.height_map = d_height_map.copy_to_host()
                    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))   

                # Reset flux array
                d_flux.copy_to_device(np.zeros_like(self.height_map, dtype=np.float64))
    
                # Launch the CUDA kernel
                compute_one_erosion_step_cuda[blocks_per_grid, threads_per_block](
                    np.float64(PARAMS["k"]), d_height_map, np.float64(self.maximum_slope),
                    d_flux, d_vhl_pos, d_vhl_mask
                )

                # Apply flux to height map
                apply_flux_to_height_map[blocks_per_grid, threads_per_block](
                    np.float64(PARAMS["k"]), d_height_map, d_flux
                )
    
                # Copy flux back to check the stopping condition
                q = d_flux.copy_to_host()
                pbar.update(1)
                #print(self.vhl.mask)
                #print(PARAMS)
                #print(np.float64(PARAMS["k"]), self.height_map, np.float64(self.maximum_slope), d_vhl_pos, sep=" == ")
                #print(q)
        # Ensure the final state of the height map is stored
        self.height_map = d_height_map.copy_to_host()


@cuda.jit
def compute_one_erosion_step_cuda(K, height_map, maximum_slope, flux, vhl_pos, vhl_mask):
    """
    CUDA kernel to perform a single step of erosion on the height map.

    Args:
        K (float): Erosion rate constant.
        height_map (numpy.ndarray): 2D array representing the terrain height.
        maximum_slope (float): Maximum allowable slope.
        flux (numpy.ndarray): Flux array to store changes in the terrain height.
        vhl_pos (tuple or None): Position of the vehicle (if any).
        vhl_mask (numpy.ndarray): Mask indicating the shape/area of the vehicle.
    """
    i, j = cuda.grid(2)
    rows, cols = height_map.shape

    # Check if within bounds
    if i >= 1 and i < rows - 1 and j >= 1 and j < cols - 1:

        def avoid_vhl(i, j):
            if vhl_pos[0] == -1:  # Check if vhl_pos is effectively None
                return False
            mask_y = int(i - vhl_pos[0])
            mask_x = int(j - vhl_pos[1])
            if (0 <= mask_y < vhl_mask.shape[0] and
                0 <= mask_x < vhl_mask.shape[1] and
                vhl_mask[mask_y, mask_x]):
                return True
            return False

        # Skip processing cells occupied by the vehicle
        if avoid_vhl(i, j):
            return

        for di, dj, dist in NEIGHBORS:
            ni, nj = int(i + di), int(j + dj)

            # Skip neighbor cells occupied by the vehicle
            if avoid_vhl(ni, nj):
                continue

            # Calculate the slope between the current cell and the neighbor
            slope = (height_map[ni, nj] - height_map[i, j]) / dist

            # If the slope exceeds the maximum allowable slope, compute flux
            if slope > maximum_slope:
                flux_val = (slope - maximum_slope)

                cuda.atomic.add(flux, (ni, nj), -flux_val)  # Reduce flux at the neighbor cell
                cuda.atomic.add(flux, (i, j), flux_val)    # Increase flux at the current cell

@cuda.jit
def apply_flux_to_height_map(K, height_map, flux):
    """
    CUDA kernel to apply the computed flux to the height map.

    Args:
        K (float): Erosion rate constant.
        height_map (numpy.ndarray): 2D array representing the terrain height.
        flux (numpy.ndarray): Flux array containing the changes to be applied.
    """
    i, j = cuda.grid(2)
    rows, cols = height_map.shape

    if i < rows and j < cols:
        height_map[i, j] += K * flux[i, j]
