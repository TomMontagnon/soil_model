from utils import *
from config import *
import numpy as np
import tqdm


class Vehicule:
    def __init__(self, mask, simulator):
        """
        Initialize a vehicule to interact with the height map.
        :param mask: A matrix representing the vehicule's mask.
        :param simulator: Instance of SandSimulator to modify the height map.
        """
        self.mask = mask
        #self.normal_map = generate_normal_map(mask)

        # Calculate the center of the vehicule's mask (Y, X coordinates)
        # The size is necessarily odd, so the center is always well-positioned.
        self.mask_center = np.array([(mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2])  # (Y, X)
        self.simulator = simulator
        self.simulator.vhl = self
        self.position = None

    def interact_with_sand(self, heap_pos):
        """
        Apply the vehicule's interaction to the height map.
        :param heap_pos: Position where the soil will be accumulated.

        ATTENTION : tmp = var[mask] affectation will copy the data.
        """
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
        """
        Determine the position for the new heap of pushed soil.
        Requires a convex mask.
        :param prev_pos: The previous position of the vehicule.
        :return: Coordinates of the new heap position.
        """
        tmp_heap_pos = self.mask_center.astype(float)
        direc = self.position - prev_pos
        if np.all(direc==0):
            return self.global_coord(tmp_heap_pos.astype(int))
        direc_normed = direc / np.linalg.norm(direc)

        # Move along the direction until outside the mask bounds
        while np.all(0 <= tmp_heap_pos + direc_normed) and \
              np.all(tmp_heap_pos + direc_normed < self.mask.shape) and \
              self.mask[tuple(tmp_heap_pos.astype(int))]:
            print(tmp_heap_pos)
            tmp_heap_pos += direc_normed
        
        heap_pos = (tmp_heap_pos - self.mask_center).astype(int)
        print(heap_pos)
        return self.global_coord(heap_pos)

    def global_coord(self, local_coords):
        """
        Convert local mask coordinates to global grid coordinates.
        :param local_coords: Tuple of local (y, x) coordinates within the mask.
        :return: Tuple of global (y, x) coordinates in the grid.
        """
        return np.array([self.position[0] + local_coords[0], self.position[1] + local_coords[1]])

    def follow_trajectory(self, trajectory):
        """
        Make the vehicule follow a trajectory.
        :param trajectory: List of tuples (x, y) representing successive positions.
        """
        self.position = trajectory[0]
        self.simulator.vehicule_trajectory.append(self.global_coord(self.mask_center))

        for prev_pos, pos in tqdm.tqdm(zip(trajectory, trajectory[1:]), desc=f"Trajectory (Total: {trajectory.shape[0]})", 
                                       unit=" checkpoints", dynamic_ncols=False,position=0):
            self.position = pos
            self.simulator.vehicule_trajectory.append(self.global_coord(self.mask_center))

            # Define the heap position and interact with the sand
            heap_pos = self.define_new_heap_pushed_position(prev_pos)
            self.interact_with_sand(heap_pos)

            # Simulate erosion after interaction
            self.simulator.simulate_erosion()
