from utils import *
from  config import PARAMS
import numpy as np


class Object:
    def __init__(self, mask, simulator):
        """
        Initialize a object to interact with the height map.
        :param mask: A matrix representing the object's mask.
        :param simulator: Instance of SandSimulator to modify the height map.
        """
        self.mask = mask
        self.normal_map = generate_normal_map(mask)


        # shape is odd, so // will give real center coordonates.
        self.mask_center = ((mask.shape[0]-1) // 2, (mask.shape[1]-1) // 2) # (Y,X)
        self.simulator = simulator
        self.simulator.obj = self
        self.position = (0,0) # TODO set None et resolve conflict

    def interact_with_sand(self, heap_pos):
        """
        Apply the object's interaction to the height map.
        """
        y, x = self.position
        h, w = self.mask.shape
        grid = self.simulator.height_map
        
        # Ensure the object stays within the grid bounds
        y_end = min(y + h, grid.shape[0])
        x_end = min(x + w, grid.shape[1])
        
        y_mask_end = min(grid.shape[0]-y,h)
        x_mask_end = min(grid.shape[1]-x,w)
        mask = self.mask[:y_mask_end,:x_mask_end]


        chunk = grid[y:y_end, x:x_end]
        chunk_upped = chunk + PARAMS["object_depth"]
        collision_zone = chunk_upped*mask
        soil_amount = np.sum(collision_zone) * PARAMS["cell_edge_length"] 
        if soil_amount < 0: # Let's flat the zone with the mean
            grid[y:y_end, x:x_end][mask == True] = np.mean(grid[y:y_end,x:x_end][mask == True])
        else: # 
            grid[y:y_end, x:x_end][mask == True] = -PARAMS["object_depth"]
            print(f"Position du tas : {heap_pos}, quantité : {soil_amount:.3f}")
            grid[heap_pos] += soil_amount

    def define_new_heap_pushed_position(self, prev_pos):
        """
        Need convex mask
        """

        direc = self.position - prev_pos
        direc_normed = direc / np.linalg.norm(direc)
        tmp = np.array(self.mask_center).astype(float)
        while 0<=tmp[0]<self.mask.shape[0] and 0<=tmp[1]<self.mask.shape[1] and self.mask[tuple(tmp.astype(int))]:
            tmp += direc_normed
        return self.global_coord(tmp.astype(int))

    def global_coord(self, local_coords):
        return self.position[0] + local_coords[0], self.position[1] + local_coords[1]

    def follow_trajectory(self, trajectory):
        """
        Make the object follow a trajectory.
        :param trajectory: List of tuples (x, y) representing successive positions.
        """
        self.position = trajectory[0]
        self.simulator.object_trajectory.append(self.global_coord(self.mask_center))
        for prev_pos, pos in zip(trajectory,trajectory[1:]):
            print(prev_pos, pos)
            self.position = pos
            self.simulator.object_trajectory.append(self.global_coord(self.mask_center))
            heap_pos = self.define_new_heap_pushed_position(prev_pos)
            self.interact_with_sand(heap_pos)
            self.simulator.simulate_erosion()
