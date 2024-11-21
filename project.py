import random
import numpy as np
import matplotlib.pyplot as plt
import noise
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import sobel, gaussian_filter, binary_erosion

DEPTH = 50

class SandSimulator:
    def __init__(self, grid_size=(1000,1000), random_seed=None, angle_of_repose=30):
        """
        Initialize the sand simulator with a height map and repose angle.
        :param grid_size: Tuple (rows, cols) for the height map size.
        :param angle_of_repose: The repose angle of the sand (in degrees).
        """
        self.tool_trajectory = []  # Store the tool's trajectory
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
        self.height_map[0,0] = -500

        
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

    def display_3d_map(self):
        """
        Render the height map in 3D along with the tool's trajectory.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Create a grid of coordinates
        x = np.arange(self.grid_size[1])
        y = np.arange(self.grid_size[0])
        x, y = np.meshgrid(x, y)

        # Plot the surface
        tmp = ax.plot_surface(x, y, self.height_map, cmap="terrain", edgecolor='k', alpha=0.8)

        # Show the colorbar
        cbar = fig.colorbar(tmp, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Hauteur', fontsize=12)

        # Plot the tool's trajectory
        if self.tool_trajectory:
            traj = np.array(self.tool_trajectory)
            ax.plot(traj[:, 1], traj[:, 0], traj[:, 2], color="red", marker='o', label="Tool Trajectory")

        ax.set_title("Height Map with Tool Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")
        plt.legend("")
        plt.show()

    def display_height_map(self):
        """
        Affiche la height-map actuelle.
        """
        plt.imshow(self.height_map, cmap="terrain", origin="upper")
        plt.colorbar(label="Hauteur")
        plt.title("Height Map - Simulateur de Sable")
        plt.show()


class Tool:
    def __init__(self, shape, simulator):
        """
        Initialize a tool to interact with the height map.
        :param shape: A matrix representing the tool's shape.
        :param simulator: Instance of SandSimulator to modify the height map.
        """
        self.shape = shape
        self.simulator = simulator
        self.position = (0, 0)  # Initial tool position

    def move(self, new_position):
        """
        Move the tool to a new position.
        :param new_position: Tuple (x, y) representing the new position.
        """
        self.position = new_position
        self.interact_with_sand()

    def interact_with_sand(self):
        """
        Apply the tool's interaction to the height map.
        """
        x, y = self.position
        rows, cols = self.shape.shape
        grid = self.simulator.height_map
        
        # Ensure the tool stays within the grid bounds
        x_end = min(x + rows, grid.shape[0])
        y_end = min(y + cols, grid.shape[1])
        
        #grid[x:x_end, y:y_end] -= 0.1*self.shape[:x_end - x, :y_end - y]
        chunk = grid[x:x_end, y:y_end]
        chunk_upped = chunk + DEPTH
        collision_zone = chunk_upped*self.shape
        soil_amount = np.sum(collision_zone) 
        if soil_amount < 0: # Let's flat the zone with the mean
            grid[x:x_end, y:y_end][self.shape == 1] = np.mean(grid[x:x_end,y:y_end][self.shape == 1])
        else: # 
            grid[x:x_end, y:y_end][self.shape == 1] = -DEPTH
            #grid[0,0] += soil_amount #TODO NEW HEAP AROUND



    def follow_trajectory(self, trajectory):
        """
        Make the tool follow a trajectory.
        :param trajectory: List of tuples (x, y) representing successive positions.
        """
        for position in trajectory:
            self.move(position)
            #self.simulator.simulate_erosion(iterations=10)  # Simulate erosion after each move



def generate_normal_map(shape, mesh):
    X=mesh[0]
    Y=mesh[1]
    # create contours 
    shape = shape.astype(int)
    eroded_shape = binary_erosion(shape)
    contours = (shape - eroded_shape).astype(bool)
    
    # Apply a Gaussian filter to smooth the shape
    smoothed_shape = gaussian_filter(shape.astype(float), sigma=2)

    # Compute the gradient of the smoothed shape
    gradient_y_shape, gradient_x_shape = np.gradient(smoothed_shape)

    # Normalize to obtain unit vectors
    length_shape = np.sqrt(gradient_x_shape**2 + gradient_y_shape**2)
    length_shape[length_shape == 0] = 1  # Avoid division by zero
    normal_x_shape = -gradient_x_shape / length_shape  # Inversion for outward direction
    normal_y_shape = -gradient_y_shape / length_shape  # Inversion for outward direction

    # Extract normal vectors only at detected contours
    normal_x_shape_contours = normal_x_shape[contours]
    normal_y_shape_contours = normal_y_shape[contours]
    X_contours = X[contours]
    Y_contours = Y[contours]

    return (X_contours,
            Y_contours,
            normal_x_shape_contours,
            normal_y_shape_contours)

def display_normal_map(shape, box):
    size=shape.shape[0]
    # Visualize the patatoid and contour normal vectors
    plt.figure(figsize=(10, 10))
    plt.imshow(shape, cmap='gray', origin='lower', extent=[-size / 2, size / 2, -size / 2, size / 2])
    plt.quiver(box[0], box[1], box[2], box[3], color='blue', scale=20)
    plt.title("Patatoid with normal vectors at contours")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

def disk_generator(radius=100):
    #TODO rename  parotut tool et shape
    size = radius*2

    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    X, Y = np.meshgrid(x, y)

    disk = ((X**2 + Y**2) <= radius**2).astype(int)
    return disk, (X,Y)

def patatoide_generator(size=50):
    scale_factor = 0.25  # Shape size
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    X, Y = np.meshgrid(x, y)

    # Create the patatoid
    patatoid = ((X**2) / (900 * scale_factor) + 
                (Y**2) / (1600 * scale_factor) +
                0.05 * np.sin(5 * X) * np.cos(5 * Y)) <= 1

    return patatoid, (X,Y)


def generate_linear_trajectory(start_point, end_point, N):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # Générer les points
    trajectory = np.linspace(start_point, end_point, N)
    
    trajectory = np.floor(trajectory).astype(int) 
    
    return trajectory

def main():
    # Initialize the simulator
    simulator = SandSimulator(grid_size=(1000, 1000), angle_of_repose=30, random_seed=0)

    # Create a tool (e.g., a bulldozer blade)
    tool_shape = disk_generator()
    tool = Tool(shape=tool_shape, simulator=simulator)

    # Define a trajectory for the tool
    trajectory = generate_linear_trajectory((100, 100), (800, 800), 10)

    # Make the tool follow the trajectory
    tool.follow_trajectory(trajectory)

    # Display the final height map in 3D
    simulator.display_3d_map()
    #simulator.display_2d_map()

def test():
    tool_shape, mesh = disk_generator(20)
    box = generate_normal_map(tool_shape, mesh)
    display_normal_map(tool_shape, box)

# Example usage
if __name__ == "__main__":
    #main()
    test()
