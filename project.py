import random
import numpy as np
import matplotlib.pyplot as plt
import noise
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import sobel, gaussian_filter, binary_erosion

DFT_OBJECT_DEPTH = 50
DFT_GRID_SIZE = (1000, 1000)
DFT_ANGLE = 30

IS_3D = False

FIG = plt.figure(figsize=(10, 7))
AX = None

class SandSimulator:
    def __init__(self, grid_size=DFT_GRID_SIZE, random_seed=None, angle_of_repose=DFT_ANGLE):
        """
        Initialize the sand simulator with a height map and repose angle.
        :param grid_size: Tuple (rows, cols) for the height map size.
        :param angle_of_repose: The repose angle of the sand (in degrees).
        """
        self.object_trajectory = []  # Store the object's trajectory
        self.grid_size = grid_size
        self.height_map = np.zeros(grid_size)  # Initialize a flat height map
        if random_seed is not None:
            print("oui")
            for x in range(grid_size[0]):
                for y in range(grid_size[1]):
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
                    self.height_map[x][y] = int(50*new_value)
        #self.height_map[400,400] = -50000000
        #self.height_map[0,0] = -500
        

        
        self.angle_of_repose = np.tan(np.radians(angle_of_repose))  # Convert to maximum slope

    def simulate_erosion(self, iterations=50):
        """
        Simulate sand erosion until reaching an equilibrium state.
        :param iterations: Number of iterations for the erosion process.
        """
        for _ in range(iterations):
            slopes_x = np.diff(self.height_map, axis=1, prepend=0)
            slopes_y = np.diff(self.height_map, axis=0, prepend=0)

            # Calculate sand flow based on slope and repose angle
            flow_x = np.where(slopes_x > self.angle_of_repose, slopes_x - self.angle_of_repose, 0)
            flow_y = np.where(slopes_y > self.angle_of_repose, slopes_y - self.angle_of_repose, 0)

            # Distribute the flow across the height map
            self.height_map[:, :-1] -= flow_x[:, :-1]
            self.height_map[:, 1:] += flow_x[:, :-1]
            self.height_map[:-1, :] -= flow_y[:-1, :]
            self.height_map[1:, :] += flow_y[:-1, :]

    def display_map(self):
        """
        Render the height map in 3D along with the object's trajectory.
        """
        global AX
        if IS_3D:
            AX = FIG.add_subplot(111, projection='3d')
    
            # Create a grid of coordinates
            x = np.arange(self.grid_size[1])
            y = np.arange(self.grid_size[0])
            x, y = np.meshgrid(x, y)
    
            # Plot the surface
            tmp = AX.plot_surface(x, y, self.height_map, cmap="terrain", edgecolor='k', alpha=0.8)
    
            # Show the colorbar
            cbar = FIG.colorbar(tmp, ax=AX, shrink=0.5, aspect=10, label='Height')
    
            # Plot the object's trajectory
            if self.object_trajectory:
                traj = np.array(self.object_trajectory)
                AX.plot(traj[:, 1], traj[:, 0], traj[:, 2], color="red", marker='o', label="Object Trajectory")
    
            AX.set_title("3D Height Map with Object Trajectory")
            AX.set_xlabel("X")
            AX.set_ylabel("Y")
            AX.set_zlabel("Height")
            plt.legend("")
        else:
            AX = FIG.add_subplot(111)
            
            im = AX.imshow(self.height_map, cmap="terrain", origin="lower")
            AX.set_xlim(0, self.grid_size[0])
            AX.set_ylim(0, self.grid_size[1])
            cbar = FIG.colorbar(im, ax=AX, label="Height")
            AX.set_title("Height Map - Simulateur de Sable")


class Object:
    def __init__(self, mask, simulator):
        """
        Initialize a object to interact with the height map.
        :param mask: A matrix representing the object's mask.
        :param simulator: Instance of SandSimulator to modify the height map.
        """
        self.mask = mask
        self.simulator = simulator
        self.position = (0, 0)  # Initial object position

    def move(self, new_position):
        """
        Move the object to a new position.
        :param new_position: Tuple (x, y) representing the new position.
        """
        self.position = new_position
        self.interact_with_sand()

    def interact_with_sand(self):
        """
        Apply the object's interaction to the height map.
        """
        x, y = self.position
        rows, cols = self.mask.shape
        grid = self.simulator.height_map
        
        # Ensure the object stays within the grid bounds
        x_end = min(x + rows, grid.shape[0])
        y_end = min(y + cols, grid.shape[1])
        
        chunk = grid[x:x_end, y:y_end]
        chunk_upped = chunk + DFT_OBJECT_DEPTH
        collision_zone = chunk_upped*self.mask
        soil_amount = np.sum(collision_zone) 
        if soil_amount < 0: # Let's flat the zone with the mean
            grid[x:x_end, y:y_end][self.mask == True] = np.mean(grid[x:x_end,y:y_end][self.mask == True])
        else: # 
            grid[x:x_end, y:y_end][self.mask == True] = -DFT_OBJECT_DEPTH
            #grid[0,0] += soil_amount #TODO NEW HEAP AROUND

    def display_mask_and_normals(self,box):
        x=self.position[0]
        y=self.position[1]
        w=self.mask.shape[0]
        h=self.mask.shape[1]
        im = AX.imshow(self.mask, origin="lower", cmap='coolwarm', extent=(x,x+w,y,y+h), alpha=(self.mask == False).astype(float))

        # Show sampled results for better visibility
        mask1 = np.random.randint(0, 10, size=len(box[0])) == 0
        if IS_3D:
            qc = AX.quiver((box[0]+x+w/2)[mask1], (box[1]+y+h/2)[mask1], np.zeros_like(box[0])[mask1],box[2][mask1], box[3][mask1], np.zeros_like(box[2])[mask1], color='blue')
        else:
            qc = AX.quiver((box[0]+x+w/2)[mask1], (box[1]+y+h/2)[mask1], box[2][mask1], box[3][mask1], color='blue')

    def follow_trajectory(self, trajectory):
        """
        Make the object follow a trajectory.
        :param trajectory: List of tuples (x, y) representing successive positions.
        """
        for position in trajectory:
            self.move(position)
            #self.simulator.simulate_erosion(iterations=10)  # Simulate erosion after each move



def generate_normal_map(obj_mask, mesh):
    X=mesh[0]
    Y=mesh[1]
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
    normal_x_mask = -gradient_x_mask / length_gradient  # Inversion for outward direction
    normal_y_mask = -gradient_y_mask / length_gradient  # Inversion for outward direction

    # Extract normal vectors only at detected contours
    normal_x_mask_contours = normal_x_mask[contours]
    normal_y_mask_contours = normal_y_mask[contours]
    X_contours = X[contours]
    Y_contours = Y[contours]

    return (X_contours,
            Y_contours,
            normal_x_mask_contours,
            normal_y_mask_contours)

def ellipse_generator(a=10, b=65):
    size = max(a,b)*2

    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    X, Y = np.meshgrid(x, y)

    ellipse = ((X**2 / a**2 + Y**2 / b**2) <= 1).astype(int)
    return ellipse, (X,Y)

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
    object_mask, mesh = ellipse_generator()
    obj = Object(mask=object_mask, simulator=simulator)

    # Define a trajectory for the object
    trajectory = generate_linear_trajectory((100, 100), (800, 800), 1)

    # Make the object follow the trajectory
    obj.follow_trajectory(trajectory)

    # Display the final height map in 3D
    simulator.display_map()
    
    box = generate_normal_map(object_mask, mesh)
    obj.display_mask_and_normals(box)
    plt.show()

def test():
    object_mask, mesh = ellipse_generator(20)

# Example usage
if __name__ == "__main__":
    main()
    #test()
