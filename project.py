import random
import numpy as np
import noise
import time
from datetime import datetime
from scipy.ndimage import gaussian_filter, binary_erosion

PARAMS = dict()
PARAMS["field_edge_length"] = 1000 # 1 meter
PARAMS["cell_edge_length"] = 2 # mm
PARAMS["grid_size"] = (PARAMS["field_edge_length"]/PARAMS["cell_edge_length"],PARAMS["field_edge_length"]/PARAMS["cell_edge_length"]) # (Ymax, Xmax)
PARAMS["k"] = PARAMS["cell_edge_length"] / 8
PARAMS["object_depth"] = 30 # mm
PARAMS["angle_of_respose"] = 30 # deg 
PARAMS["erosion_threshold"] = 10

NEIGHBORS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))]

IS_REGISTERED = True
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200, threshold=np.inf)

class SandSimulator:
    def __init__(self, grid_size=PARAMS["grid_size"], random_seed=None):
        """
        Initialize the sand simulator with a height map and repose angle.
        :param grid_size: Tuple (rows, cols) for the height map size.
        :param random_seed: Integer for the random seed (if None, the soil will be flat)
        """
        self.object_trajectory = []  # Store the object's trajectory
        self.grid_size = grid_size
        self.height_map = np.zeros(grid_size)  # Initialize a flat height map
        if random_seed is not None:
            for y in range(grid_size[0]):
                for x in range(grid_size[1]):
                    new_value = noise.snoise2(
                          x/1000,
                          y/1000,
                          octaves=3,
                          persistence=0.95,
                          lacunarity=2,
                          repeatx=grid_size[1],
                          repeaty=grid_size[0],
                          base=random_seed
                         )
                    self.height_map[y][x] = int(50*new_value)
        

        
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
        q = self.compute_one_erosion_step()
        nb_iteration = 1
        print("Erosion en cours...")
        while np.sum(np.abs(q)) > PARAMS["erosion_threshold"]:
            q = self.compute_one_erosion_step()
            nb_iteration+=1
            print("X",end="", flush=True)
        print("\nNombre d'iterations d'érosion: ", nb_iteration) #TODO setup tqdm

class Object:
    def __init__(self, mask, mesh, simulator):
        """
        Initialize a object to interact with the height map.
        :param mask: A matrix representing the object's mask.
        :param simulator: Instance of SandSimulator to modify the height map.
        """
        self.mask = mask
        self.normal_map = generate_normal_map(mask, mesh)


        # shape is odd, so // will give real center coordonates.
        self.mask_center = ((mask.shape[0]-1) / 2, (mask.shape[1]-1) / 2) # (Y,X)
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
        
        chunk = grid[y:y_end, x:x_end]
        chunk_upped = chunk + PARAMS["object_depth"]
        collision_zone = chunk_upped*self.mask
        soil_amount = np.sum(collision_zone) 
        if soil_amount < 0: # Let's flat the zone with the mean
            grid[y:y_end, x:x_end][self.mask == True] = np.mean(grid[y:y_end,x:x_end][self.mask == True])
        else: # 
            grid[y:y_end, x:x_end][self.mask == True] = -PARAMS["object_depth"]
            print(f"Position du tas : {heap_pos}, quantité : {soil_amount}")
            grid[heap_pos] += soil_amount * PARAMS["cell_edge_length"]

    def define_new_heap_pushed_position(self, prev_pos):
        """
        Need convex mask
        """
        direc = self.position - prev_pos
        direc_normed = direc / min(direc)
        tmp = np.array(self.mask_center)
        while tmp[0]<self.mask.shape[0] and tmp[1]<self.mask.shape[1] and self.mask[tuple(tmp.astype(int))]:
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
            self.position = pos
            self.simulator.object_trajectory.append(self.global_coord(self.mask_center))
            heap_pos = self.define_new_heap_pushed_position(prev_pos)
            self.interact_with_sand(heap_pos)
            self.simulator.simulate_erosion()



def generate_normal_map(obj_mask, mesh):
    Y_meshed=mesh[0]
    X_meshed=mesh[1]
    # create contours 
    mask = obj_mask.astype(int)
    eroded_mask = binary_erosion(mask)
    contours = (mask - eroded_mask).astype(bool)
    
    # Apply a Gaussian filter to smooth the mask
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=2)

    # Compute the gradient of the smoothed mask
    gradient_y_mask, gradient_x_mask = np.gradient(smoothed_mask)

    # Normalize to obtain unit vectors
    length_gradient = np.sqrt(gradient_x_mask**2 + gradient_y_mask**2)
    length_gradient[length_gradient == 0] = 1  # Avoid division by zero
    normal_y_mask = -gradient_y_mask / length_gradient  # Inversion for outward direction
    normal_x_mask = -gradient_x_mask / length_gradient  # Inversion for outward direction

    # Extract normal vectors only at detected contours
    normal_y_mask_contours = normal_y_mask[contours]
    normal_x_mask_contours = normal_x_mask[contours]
    Y_contours = Y_meshed[contours]
    X_contours = X_meshed[contours]
    return (Y_contours,
            X_contours,
            normal_y_mask_contours,
            normal_x_mask_contours)

def ellipse_generator(a=50, b=40):
    a=a//PARAMS["cell_edge_length"]
    b=b//PARAMS["cell_edge_length"]
    size = max(a,b)*2 
    # size is even, so divisible by 2.
    # size + 1 is odd, so the center have integers coordonates.
    # on top of that, linspaces with those constraint, generate integers.
    x = np.linspace(-size / 2, size / 2, size+1).astype(int)
    y = np.linspace(-size / 2, size / 2, size+1).astype(int)
    
    X_meshed, Y_meshed = np.meshgrid(x, y)
    ellipse = ((X_meshed**2 / a**2 + Y_meshed**2 / b**2) <= 1)
    return ellipse, (Y_meshed, X_meshed)

def generate_linear_trajectory(start_point, end_point, N):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # Générer les points
    trajectory = np.linspace(start_point, end_point, N)
    
    trajectory = np.floor(trajectory).astype(int) 
    
    return trajectory

def main():
    global simulator, obj, name

    name = "traj"

    # Initialize the simulator
    simulator = SandSimulator(random_seed=42)

    # Create a object (e.g., a bulldozer blade)
    mask, mesh = ellipse_generator(10,15)
    obj = Object(mask=mask, mesh=mesh, simulator=simulator)

    # Define a trajectory for the object
    traj1 = generate_linear_trajectory((50, 50), (150, 150), 2)
    traj2 = generate_linear_trajectory((50, 150), (150, 50), 8)
    traj = np.concatenate((traj1,traj2))
    # Make the object follow the trajectory
    obj.follow_trajectory(traj)

def test():
    global simulator, obj, name

    name = "test"
    # TODO Test scalability of the simulation
    simulator = SandSimulator(grid_size = (20,20))
    simulator.height_map[10,10] = 1000
    mask, mesh = ellipse_generator(4,4)
    obj = Object(mask=mask, mesh=mesh, simulator=simulator)
    
    simulator.simulate_erosion()


simulator = None
obj = None
name = None


# Example usage
if __name__ == "__main__":
    start = time.time()

    #main()
    test()

    time = time.time() - start
    print(f"Temps d'exécution : {time:.3f} secondes")
    filename = f"{name} {datetime.now().strftime("%Y-%m-%d %H:%M")} ({time:.3f}s)"
    if IS_REGISTERED:
        np.savez_compressed("data/"+filename,
                        height_map=simulator.height_map,
                        traj=np.array(simulator.object_trajectory),
                        obj_mask=obj.mask,
                        obj_pos=obj.position,
                        obj_normal_map=obj.normal_map,
                        time=time) # TODO register PARAMS

