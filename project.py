import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SandSimulator:
    def __init__(self, grid_size, angle_of_repose):
        """
        Initialize the sand simulator with a height map and repose angle.
        :param grid_size: Tuple (rows, cols) for the height map size.
        :param angle_of_repose: The repose angle of the sand (in degrees).
        """
        self.grid_size = grid_size
        self.angle_of_repose = np.tan(np.radians(angle_of_repose))  # Convert to maximum slope
        self.height_map = np.zeros(grid_size)  # Initialize a flat height map
        self.tool_trajectory = []  # Store the tool's trajectory
    
        self.height_map[0,0] = 1

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
        plt.legend()
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
        
        grid[x:x_end, y:y_end] -= 0.1*self.shape[:x_end - x, :y_end - y]

    def follow_trajectory(self, trajectory):
        """
        Make the tool follow a trajectory.
        :param trajectory: List of tuples (x, y) representing successive positions.
        """
        for position in trajectory:
            self.move(position)
            #self.simulator.simulate_erosion(iterations=10)  # Simulate erosion after each move


def circle_generator(radius=100):
    diameter = radius * 2
    tool_shape = np.zeros((diameter, diameter), dtype=int)
    
    # Generate the circle
    for y in range(diameter):
        for x in range(diameter):
            if (x - radius)**2 + (y - radius)**2 <= radius**2:
                tool_shape[y, x] = 1
    return tool_shape

def generate_linear_trajectory(start_point, end_point, N):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # Générer les points
    trajectory = np.linspace(start_point, end_point, N)
    
    trajectory = np.floor(trajectory).astype(int) 
    
    return trajectory

# Example usage
if __name__ == "__main__":

    # Initialize the simulator
    simulator = SandSimulator(grid_size=(1000, 1000), angle_of_repose=30)

    # Create a tool (e.g., a bulldozer blade)
    tool_shape = circle_generator()
    tool = Tool(shape=tool_shape, simulator=simulator)

    # Define a trajectory for the tool
    trajectory = generate_linear_trajectory((100,100), ( 800,300), 50)

    # Make the tool follow the trajectory
    tool.follow_trajectory(trajectory)

    # Display the final height map in 3D
    simulator.display_3d_map()
    #simulator.display_2d_map()

