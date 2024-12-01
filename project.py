import random
import numpy as np
import matplotlib.pyplot as plt
import noise
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, binary_erosion

DFT_OBJECT_DEPTH = 30
DFT_GRID_SIZE = (200, 200) # (Ymax, Xmax)

ANGLE_OF_RESPOSE = 30
DX = 5 #mm
K = DX / (16)

IS_3D = True
FIG = plt.figure(figsize=(10, 7))
AX = None


np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200, threshold=np.inf)




class SandSimulator:
    def __init__(self, grid_size=DFT_GRID_SIZE, random_seed=None):
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
                          repeatx=grid_size[0],
                          repeaty=grid_size[1],
                          base=random_seed
                         )
                    self.height_map[y][x] = int(50*new_value)
        

        
        self.maximum_slope = np.tan(np.radians(ANGLE_OF_RESPOSE))  # Convert to maximum slope

    def compute_one_erosion_step(self):
        """
        Simulates one erosion step on the height map and calculates q (flux).
        """
        global DX, K

        # Define the 8-neighbor coordinates
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        # Initialize the q array to store the flux between grid points
        q = np.zeros_like(self.height_map)

        # Get the dimensions of the height map
        rows, cols = self.height_map.shape

        # Iterate through each cell of the height map
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                
                mask_coord = tuple([i-self.obj.position[0], j-self.obj.position[1]])
                if 0 <= mask_coord[0] < self.obj.mask.shape[0] and \
                    0 <= mask_coord[1] < self.obj.mask.shape[1] and \
                    self.obj.mask[mask_coord]:
                    continue

                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    
                    mask_coord = tuple([ni-self.obj.position[0], nj-self.obj.position[1]])
                    if 0 <= mask_coord[0] < self.obj.mask.shape[0] and \
                        0 <= mask_coord[1] < self.obj.mask.shape[1] and \
                        self.obj.mask[mask_coord]:
                        continue

                    # Compute the slope between the current cell and the neighbor
                    slope = (self.height_map[ni, nj] - self.height_map[i, j]) / DX

                    # Calculate the flux based on the slope and update q
                    if slope < -self.maximum_slope:
                        flux = K  * (slope + self.maximum_slope)
                        q[ni, nj] -= flux
                        q[i, j] += flux
                    elif slope > self.maximum_slope:
                        flux = K  * (slope - self.maximum_slope)
                        q[ni, nj] -= flux
                        q[i, j] += flux
        self.height_map += q 
        return q
    
    def simulate_erosion(self):
        q = self.compute_one_erosion_step()
        nb_iteration = 1
        print("Erosion en cours...")
        while np.sum(np.abs(q)) > 10:
            q = self.compute_one_erosion_step()
            nb_iteration+=1
            print("X",end="", flush=True)
        print("\nNombre d'iterations d'érosion: ", nb_iteration)

    def display_map(self):
        """
        Render the height map in 3D along with the object's trajectory.
        """
        global AX
        if IS_3D:
            AX = FIG.add_subplot(111, projection='3d')
    
            # Create a grid of coordinates
            y = np.arange(self.grid_size[0])
            x = np.arange(self.grid_size[1])
            x_meshed, y_meshed = np.meshgrid(x, y)
    
            # Plot the surface
            tmp = AX.plot_surface(x_meshed, y_meshed, self.height_map, cmap="terrain", edgecolor='k', alpha=0.4)
    
            # Show the colorbar
            cbar = FIG.colorbar(tmp, ax=AX, shrink=0.5, aspect=10, label='Height')
    
            # Plot the object's trajectory
            if self.object_trajectory:
                traj = np.array(self.object_trajectory)
                AX.plot(traj[:, 1], traj[:, 0], np.full(traj[:,0].shape,np.min(self.height_map)), color="red", marker='o', label="Object Trajectory")
    
            AX.set_title("3D Height Map with Object Trajectory")
            AX.set_zlabel("Height")
        else:
            AX = FIG.add_subplot(111)
            
            im = AX.imshow(self.height_map, cmap="terrain", origin="lower")
            cbar = FIG.colorbar(im, ax=AX, label="Height")
            AX.set_title("Height Map - Simulateur de Sable")

            # Plot the object's trajectory
            if self.object_trajectory:
                traj = np.array(self.object_trajectory)
                AX.plot(traj[:, 1], traj[:, 0], color="red", marker='o', label="Object Trajectory")
        AX.set_ylim(0, self.grid_size[0])
        AX.set_xlim(0, self.grid_size[1])
        AX.set_xlabel("X")
        AX.set_ylabel("Y")
        plt.legend("")


class Object:
    def __init__(self, mask, simulator):
        """
        Initialize a object to interact with the height map.
        :param mask: A matrix representing the object's mask.
        :param simulator: Instance of SandSimulator to modify the height map.
        """
        self.mask = mask
        # shape is odd, so // will give real center coordonates.
        self.mask_center = ((mask.shape[0]-1) / 2, (mask.shape[1]-1) / 2) # (Y,X)
        self.simulator = simulator
        self.simulator.obj = self
        self.position = (0, 0)  # (Y,X)

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
        chunk_upped = chunk + DFT_OBJECT_DEPTH
        collision_zone = chunk_upped*self.mask
        soil_amount = np.sum(collision_zone) 
        if soil_amount < 0: # Let's flat the zone with the mean
            grid[y:y_end, x:x_end][self.mask == True] = np.mean(grid[y:y_end,x:x_end][self.mask == True])
        else: # 
            grid[y:y_end, x:x_end][self.mask == True] = -DFT_OBJECT_DEPTH
            print(f"Position du tas : {heap_pos}, quantité : {soil_amount}")
            grid[heap_pos] += soil_amount

    def display_mask(self):
        y=self.position[0]
        x=self.position[1]
        h=self.mask.shape[0]
        w=self.mask.shape[1]
        im = AX.imshow(self.mask, origin="lower", cmap='coolwarm', extent=(x,x+w,y,y+h), alpha=self.mask.astype(float)) 

    def display_normals(self, normal_map):
        concat = np.column_stack(normal_map[0:2])
        transformed = np.apply_along_axis(lambda coord: self.global_coord(coord + self.mask_center),1,concat)
        Y = transformed[:,0]
        X = transformed[:,1]
        V = normal_map[2]
        U = normal_map[3]

        # Show sampled results for better visibility
        mask1 = np.random.randint(0, 4, size=len(normal_map[0])) == 0
        if IS_3D:
            Z = np.zeros_like(X)
            W = np.zeros_like(U)
            qc = AX.quiver(X[mask1], Y[mask1], Z[mask1], U[mask1], V[mask1], W[mask1], color='blue',length=20)
        else:
            qc = AX.quiver(X[mask1], Y[mask1], U[mask1], V[mask1], color='blue', length=20)

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

def ellipse_generator(a=35, b=25):
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
    # Initialize the simulator
    simulator = SandSimulator(random_seed=42)

    # Create a object (e.g., a bulldozer blade)
    object_mask, mesh = ellipse_generator(10,15)
    obj = Object(mask=object_mask, simulator=simulator)

    # Define a trajectory for the object
    traj1 = generate_linear_trajectory((50, 50), (150, 150), 8)
    traj2 = generate_linear_trajectory((50, 150), (150, 50), 8)
    traj = np.concatenate((traj1,traj2))
    print(traj)
    # Make the object follow the trajectory
    obj.follow_trajectory(traj)

    # Display the final height map in 3D
    simulator.display_map()
    #normal_map = generate_normal_map(object_mask, mesh)
    obj.display_mask()
    #obj.display_normals(normal_map)

def test():
    simulator = SandSimulator()
    simulator.height_map[10,10] = 1000
    object_mask, mesh = ellipse_generator(3,3)
    obj = Object(mask=object_mask, simulator=simulator)
    
    simulator.simulate_erosion()

    simulator.display_map()
    obj.display_mask()



# Example usage
if __name__ == "__main__":
    start = time.time()

    main()
    #test()

    end = time.time()
    print(f"Temps d'exécution : {end-start:.3f} secondes")
    plt.show()

