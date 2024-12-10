from utils import *
from config import PARAMS
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
        self.normal_map = generate_normal_map(mask)

        # Calculate the center of the vehicule's mask (Y, X coordinates)
        self.mask_center = ((mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2)  # (Y, X)
        self.simulator = simulator
        self.simulator.vhl = self
        self.position = None

    def interact_with_sand(self, heap_pos):
        """
        Apply the vehicule's interaction to the height map.
        :param heap_pos: Position where the soil will be accumulated.
        """
        y, x = self.position
        h, w = self.mask.shape
        grid = self.simulator.height_map

        # Ensure the vehicule stays within the grid bounds
        y_end = min(y + h, grid.shape[0])
        x_end = min(x + w, grid.shape[1])

        y_mask_end = min(grid.shape[0] - y, h)
        x_mask_end = min(grid.shape[1] - x, w)
        mask = self.mask[:y_mask_end, :x_mask_end]

        # Interaction with the height map
        chunk = grid[y:y_end, x:x_end]
        chunk_upped = chunk + PARAMS["vehicule_depth"]
        collision_zone = chunk_upped * mask
        soil_amount = np.sum(collision_zone)

        if soil_amount < 0:  # Flatten the zone with the mean
            grid[y:y_end, x:x_end][mask == True] = np.mean(grid[y:y_end, x:x_end][mask == True])
        else:  # Push soil into the heap
            grid[y:y_end, x:x_end][mask == True] = -PARAMS["vehicule_depth"]
            #print(f"Heap position: {heap_pos}, quantity: {soil_amount:.3f}")
            grid[heap_pos] += soil_amount

    def define_new_heap_pushed_position(self, prev_pos):
        """
        Determine the position for the new heap of pushed soil.
        Requires a convex mask.
        :param prev_pos: The previous position of the vehicule.
        :return: Coordinates of the new heap position.
        """
        direc = self.position - prev_pos
        direc_normed = direc / np.linalg.norm(direc)
        tmp = np.array(self.mask_center).astype(float)

        # Move along the direction until outside the mask bounds
        while 0 <= tmp[0] < self.mask.shape[0] and 0 <= tmp[1] < self.mask.shape[1] and self.mask[tuple(tmp.astype(int))]:
            tmp += direc_normed

        return self.global_coord(tmp.astype(int))

    def global_coord(self, local_coords):
        """
        Convert local mask coordinates to global grid coordinates.
        :param local_coords: Tuple of local (y, x) coordinates within the mask.
        :return: Tuple of global (y, x) coordinates in the grid.
        """
        return self.position[0] + local_coords[0], self.position[1] + local_coords[1]

    def follow_trajectory(self, trajectory):
        """
        Make the vehicule follow a trajectory.
        :param trajectory: List of tuples (x, y) representing successive positions.
        """
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
