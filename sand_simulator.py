import numpy as np
import tqdm
from config import *
from utils import *
from numba import jit, cuda, float64
from os import system

# Define the neighborhood for erosion simulation (dx, dy, distance)
NEIGHBORS = np.array([
    (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), 
    (-1, -1,np.sqrt(2)), (1, 1,np.sqrt(2)), 
    (-1, 1,np.sqrt(2)), (1, -1,np.sqrt(2))
], dtype=np.float64)

class SandSimulator:
    def __init__(self, grid_size, use_cuda=True, random_seed=None):
        """Initialize terrain height map and repose angle."""

        self.grid_size = grid_size
        self.vhl = None
        self.vehicule_trajectory = [] # Store the real trajectory done

        self.use_cuda = use_cuda
        if random_seed is None:
            self.height_map = np.zeros(grid_size)  # Flat terrain
        else:
            self.height_map = generate_nonflat_field(grid_size, random_seed)  # Random terrain
            
        # maximum slope based on the repose angle
        self.maximum_slope = np.tan(np.radians(PARAMS["angle_of_respose"]))
       
        # register initial state
        HM_STATES.append((self.height_map.copy(), np.array(self.vehicule_trajectory).copy()))

    def final_erosion(self):
        """Run final erosion simulation without vehicule"""

        self.vhl.position = None 
        self.simulate_erosion(register=True)


    def simulate_erosion(self, register=False):
        """Call the right erosion function"""
        if self.use_cuda:
            self.simulate_erosion_cuda(register)
        else:
            self.simulate_erosion_jit(register)

    def simulate_erosion_cuda(self, register):
        """Perform erosion using CUDA for GPU acceleration."""

        # Transfer data to GPU
        d_height_map = cuda.to_device(self.height_map)
        d_flux = cuda.to_device(np.zeros_like(self.height_map, dtype=np.float64))
        d_vhl_mask = cuda.to_device(self.vhl.mask)
        d_vhl_pos = cuda.to_device(np.array(self.vhl.position or (-1, -1), dtype=np.int32))
    
        # Define thread and block configuration
        threads = (32, 32)  # Thread block size
        blocks = tuple((dim + threads[i] - 1) // threads[i] for i, dim in enumerate(self.height_map.shape))

        # Initialize a progress bar to track the erosion process
        with tqdm.tqdm(desc="Eroding", unit=" erosions", position=1, leave=False) as pbar:
            q = np.inf  # Initialize with a large value
            while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
            #for i in range(500):
                
                # Reset flux array
                d_flux.copy_to_device(np.zeros_like(self.height_map, dtype=np.float64))
    
                # Launch the CUDA kernel
                compute_one_erosion_step_cuda[blocks, threads](
                    np.float64(PARAMS["k"]), d_height_map, np.float64(self.maximum_slope),
                    d_flux, d_vhl_pos, d_vhl_mask
                )

                # Apply flux to height map
                apply_flux_to_height_map[blocks, threads](
                    np.float64(PARAMS["k"]), d_height_map, d_flux
                )
                
                #register
                if register:
                    tmp_hm = d_height_map.copy_to_host()
                    HM_STATES.append((tmp_hm.copy(), np.array(self.vehicule_trajectory).copy()))   
    
                # Copy flux back to check the stopping condition
                q = d_flux.copy_to_host()
                pbar.update(1)

        # Ensure the final state of the height map is stored
        self.height_map = d_height_map.copy_to_host()

    def simulate_erosion_jit(self, register):
        """Perform erosion using JIT acceleration."""

        with tqdm.tqdm(desc="Eroding", unit=" erosions", position=1,leave=False) as pbar:
            q = np.inf  # Initialize with a large value
            while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
                q = compute_one_erosion_step_jit(np.float64(PARAMS["k"]), self.height_map, 
                                             self.maximum_slope, self.vhl.position, self.vhl.mask)
                if register:
                    HM_STATES.append((self.height_map.copy(), np.array(self.vehicule_trajectory).copy()))   
                pbar.update(1)



@cuda.jit
def compute_one_erosion_step_cuda(K, height_map, maximum_slope, flux, vhl_pos, vhl_mask):
    """CUDA kernel: Simulates a single erosion step on the height map and calculates the flux (q) between cells"""

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
    """CUDA kernel: Update height map using erosion flux."""

    i, j = cuda.grid(2)
    rows, cols = height_map.shape

    if i < rows and j < cols:
        height_map[i, j] += K * flux[i, j]


    

@jit(nopython=True, parallel=True)
def compute_one_erosion_step_jit(K, height_map, maximum_slope, vhl_pos, vhl_mask):
    """With JIT : Simulates a single erosion step on the height map and calculates the flux (q) between cells"""
    def avoid_vhl(i, j):
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

