from utils import *
from config import *
import numpy as np
import tqdm


class Vehicule:
    def __init__(self, mask, simulator):
        """Initialize the vehicle with a mask and link it to the simulator."""
        global VHL_MASK
        self.mask = VHL_MASK = mask
        #self.normal_map = generate_normal_map(mask)

        # Calculate the center of the vehicule's mask (Y, X coordinates)
        self.mask_center = ((mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2)  # (Y, X)
        self.simulator = simulator
        self.simulator.vhl = self
        self.position = None  


    def interact_with_sand(self, heap_pos):
        """Interact with the sand height map at the current position, displacing soil."""
        y, x = self.position
        h, w = self.mask.shape
        grid = self.simulator.height_map

        # Ensure the vehicule stays within the grid bounds
        y_end = min(y + h, grid.shape[0])
        x_end = min(x + w, grid.shape[1])

        y_end_mask = min(grid.shape[0] - y, h)
        x_end_mask = min(grid.shape[1] - x, w)
        mask = self.mask[:y_end_mask, :x_end_mask]

        # Interaction with the height map
        chunk = grid[y:y_end, x:x_end]
        chunk_saved = chunk[mask].copy()

        chunk[mask] = np.minimum(chunk[mask], - PARAMS["vehicule_depth"])

        soil_amount = np.sum(np.abs(chunk_saved - chunk[mask]))
        grid[heap_pos] += soil_amount


    def define_new_heap_pushed_position(self, prev_pos):
        """Define the position to push the displaced soil based on movement direction."""
        tmp_heap_pos = np.array(self.mask_center).astype(float)
        direc = self.position - prev_pos
        if np.all(direc==0):
            return self.global_coord(tmp_heap_pos.astype(int))
        direc_normed = direc / np.linalg.norm(direc)

        # Move along the direction until outside the mask bounds
        while np.all(0 <= tmp_heap_pos + direc_normed) and \
              np.all(tmp_heap_pos + direc_normed < self.mask.shape) and \
              self.mask[tuple(tmp_heap_pos.astype(int))]:
            tmp_heap_pos += direc_normed
        
        heap_pos = self.global_coord(tmp_heap_pos.astype(int))
        heap_pos = tuple(np.minimum(heap_pos, np.array(self.simulator.grid_size) -1))
        return heap_pos
    

    def global_coord(self, local_coords):
        """Convert local coordinates within the mask to global grid coordinates."""
        return self.position[0] + local_coords[0], self.position[1] + local_coords[1]

    def follow_trajectory(self, trajectory):
        """Make the vehicle follow a trajectory and interact with the sand at each step."""
        leng = trajectory.shape[0]
        self.position = trajectory[0]
        self.simulator.vehicule_trajectory.append(self.global_coord(self.mask_center))

        for prev_pos, pos in tqdm.tqdm(zip(trajectory, trajectory[1:]), desc=f"Trajectory (Total: {leng})", unit=" checkpoints", dynamic_ncols=False,position=0):
            self.position = pos
            self.simulator.vehicule_trajectory.append(self.global_coord(self.mask_center))

            # Define the heap position and interact with the sand
            heap_pos = self.define_new_heap_pushed_position(prev_pos)
            self.interact_with_sand(heap_pos)

            # Simulate erosion after interaction
            self.simulator.simulate_erosion()

