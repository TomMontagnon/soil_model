from scipy.ndimage import gaussian_filter, binary_erosion
from noise import snoise2
import numpy as np
import open3d as o3d

def generate_normal_map(vhl_mask):
    """
    Generates the normal vectors for a vehicule's mask.
    :param vhl_mask: 2D array representing the vehicule mask.
    :return: Tuple containing contour coordinates and normal vectors (Y, X, normal_y, normal_x).
    """
    X_meshed, Y_meshed = generate_meshgrid(vhl_mask.shape[0]) 

    # Create contours from the vehicule mask
    mask = vhl_mask.astype(int)
    eroded_mask = binary_erosion(mask)
    contours = (mask - eroded_mask).astype(bool)
    
    # Smooth the mask using a Gaussian filter and compute gradients
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=2)
    gradient_y_mask, gradient_x_mask = np.gradient(smoothed_mask)

    # Normalize gradients to obtain unit vectors
    length_gradient = np.sqrt(gradient_x_mask**2 + gradient_y_mask**2)
    length_gradient[length_gradient == 0] = 1  # Avoid division by zero
    normal_y_mask = -gradient_y_mask / length_gradient
    normal_x_mask = -gradient_x_mask / length_gradient

    # Extract normal vectors at the detected contours
    normal_y_mask_contours = normal_y_mask[contours]
    normal_x_mask_contours = normal_x_mask[contours]
    Y_contours = Y_meshed[contours]
    X_contours = X_meshed[contours]
    return (Y_contours, X_contours, normal_y_mask_contours, normal_x_mask_contours)

def ellipse_generator(a, b):
    """
    Generates an elliptical mask with specified semi-major (a) and semi-minor (b) axes.
    :param a: Semi-major axis length.
    :param b: Semi-minor axis length.
    :return: 2D binary array representing the ellipse.
    """
    if a==0:
        a=1
    if b==0:
        b=1

    # Ensure the size is even to center the ellipse on integer coordinates
    size = max(a, b) * 2 + 1

    X_meshed, Y_meshed = generate_meshgrid(size)

    ellipse = ((X_meshed**2 / a**2 + Y_meshed**2 / b**2) <= 1)
    return ellipse

def generate_meshgrid(size):
    """
    Generates a 2D meshgrid of specified size.
    :param size: Integer defining the grid dimensions.
    :return: Two 2D arrays for the X and Y coordinates of the grid.
    """
    x = np.linspace(int(-size / 2), int(size / 2), size).astype(int)
    y = np.linspace(int(-size / 2), int(size / 2), size).astype(int)
    X_meshed, Y_meshed = np.meshgrid(x, y)
    return X_meshed, Y_meshed

def generate_linear_trajectory(start_point, end_point, N):
    """
    Generates a linear trajectory between two points.
    :param start_point: Tuple (x, y) for the starting point.
    :param end_point: Tuple (x, y) for the ending point.
    :param N: Number of points in the trajectory.
    :return: 2D array of integer coordinates along the trajectory.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    trajectory = np.linspace(start_point, end_point, N)

    trajectory = np.floor(trajectory).astype(int) 
    
    return trajectory

def generate_sinusoidal_trajectory(start_point, end_point, frequency, amplitude, N=100):
    """
    Generate a sinusoidal traj; between two points.
    
    :param start_point: Tuple (x, y) for the start of the traj.
    :param end_point: Tuple (x, y) for the end of the traj.
    :param frequency: Frequency of the sinusoidal wave.
    :param amplitude: Amplitude of the sinusoidal wave.
    :param N: Number of points to generate along the traj.
    :return: Two arrays (x, y) representing the coordinates of the sinusoidal traj.
    """
    # Convert points to numpy arrays
    start = np.array(start_point)
    end = np.array(end_point)
    
    # Generate linearly spaced points along the straight line
    t = np.linspace(0, 1, N)
    linear_path = (1 - t)[:, None] * start + t[:, None] * end
    
    # Calculate direction vector and perpendicular vector
    direction = end - start
    length = np.linalg.norm(direction)
    direction_unit = direction / length
    perpendicular = np.array([-direction_unit[1], direction_unit[0]])  # Rotate by 90 degrees
    
    # Generate sinusoidal offsets
    sinusoidal_offsets = amplitude * np.sin(2 * np.pi * frequency * t)
    sinusoidal_offsets_2D = perpendicular[None, :] * sinusoidal_offsets[:, None]
    
    # Add sinusoidal offsets to the linear path
    sinusoidal_path = linear_path + sinusoidal_offsets_2D
    
    return np.floor(sinusoidal_path).astype(int)


def generate_nonflat_field(grid_size, random_seed):
    """
    Generates a random non-flat terrain using Perlin noise.
    :param grid_size: Tuple (rows, cols) specifying the size of the grid.
    :param random_seed: Integer seed for deterministic noise generation.
    :return: 2D array representing the terrain height map.
    """
    height_map = np.zeros(grid_size)
    
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            new_value = snoise2(
                x / grid_size[1],
                y / grid_size[0],
                octaves=3,
                persistence=0.95,
                lacunarity=2,
                repeatx=grid_size[1],
                repeaty=grid_size[0],
                base=random_seed
            )
            height_map[y][x] = int(50 * new_value)  # Scale the noise value
        
    return height_map



def point_array_to_heightmap(pts, shape, cell_size):
    y_bins, x_bins = shape
    heightmap = np.full((y_bins, x_bins), np.nan, dtype=np.float32)
    counts = np.zeros((y_bins, x_bins), dtype=np.int32)
    
    # Accumulate heights for each grid cell
    for x, y, z in pts:
        x_idx = int(x / cell_size)
        y_idx = int(y / cell_size)
        if not (0 <= x_idx < x_bins and 0 <= y_idx < y_bins):
            raise ValueError("Generate cell coordonates out of grid.")
        if np.isnan(heightmap[y_idx, x_idx]):
            heightmap[y_idx, x_idx] = z
        else:
            heightmap[y_idx, x_idx] += z
        counts[y_idx, x_idx] += 1

    # Average heights
    valid_cells = counts > 0
    if np.any(valid_cells):
        heightmap[valid_cells] /= counts[valid_cells]
    else:
        raise ValueError("No valid points found for heightmap generation.")
    return heightmap

def point_clouds_to_heightmap(pcds, cell_size):
    """Convert an Open3D point cloud to a heightmap (NumPy array) by averaging point heights within each grid cell."""


    padding = 1
    points = []
    for pcd in pcds:
        points.append(np.asarray(pcd.points))

        if points[-1].size == 0:
            raise ValueError("Input point cloud is empty.")

    x_min = np.min([np.min(pts[:, 0]) for pts in points]) - padding
    x_max = np.max([np.max(pts[:, 0]) for pts in points]) + padding
    y_min = np.min([np.min(pts[:, 1]) for pts in points]) - padding
    y_max = np.max([np.max(pts[:, 1]) for pts in points]) + padding

    points = [pts - np.array([x_min, y_min, 0]) for pts in points]

    # Define grid size
    x_bins = int((x_max - x_min) / cell_size) + 1
    y_bins = int((y_max - y_min) / cell_size) + 1
    if x_bins < 10 or y_bins < 10:
        raise ValueError(f"Too small dimensions : ({y_bins}, {x_bins}).")

    heightmaps = [point_array_to_heightmap(pts, (y_bins, x_bins), cell_size) for pts in points]
    return heightmaps, (y_bins, x_bins)


def ros_odom_to_trajectory(ros_odom_array, shape, nb_checkpoints):
    return np.zeros((nb_checkpoints, 2),np.nan)


