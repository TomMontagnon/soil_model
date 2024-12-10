import numpy as np
import time
from datetime import datetime
import tqdm
from utils import *
from config import PARAMS
from sand_simulator import SandSimulator
from vehicule import Vehicule

IS_REGISTERED = True
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200, threshold=np.inf)

HM_STATES = list()
VHL_MASK = None  
NAME = None

#TODO better display of tqdm, fix unit/s

def main():
    """
    Main function to initialize the sand simulator and simulate vehicule movement
    along a defined trajectory.
    """
    global HM_STATES, VHL_MASK, NAME
    NAME = "traj X"

    # Initialize the simulator
    simulator = SandSimulator()
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))

    # Create a vehicule (e.g., a bulldozer blade)
    VHL_MASK = ellipse_generator(30 // PARAMS["cell_edge_length"], 50 // PARAMS["cell_edge_length"])
    vhl = Vehicule(mask=VHL_MASK, simulator=simulator)

    # Define a trajectory for the vehicule
    traj_comp = list()
    #traj_comp.append(generate_linear_trajectory((100, 900), (900, 100), 1))
    traj_comp.append(generate_sinusoidal_trajectory((100,100), (700,700), 2, 100, 25))
    traj = np.concatenate(traj_comp) // PARAMS["cell_edge_length"] - np.array(vhl.mask_center)

    # Make the vehicule follow the trajectory
    vhl.follow_trajectory(traj)
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))

def test():
    """
    Test function to initialize the sand simulator and simulate basic vehicule
    interactions with the height map.
    """
    global HM_STATES, VHL_MASK, NAME

    NAME = "test"

    # Initialize a small sand simulator grid
    simulator = SandSimulator()
    simulator.height_map[10, 10] = 1000  # Modify height map for testing
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))

    # Create a simple vehicule mask
    VHL_MASK = ellipse_generator(10, 10)
    vhl = Vehicule(mask=VHL_MASK, simulator=simulator)

    # Simulate erosion
    simulator.simulate_erosion()
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))

# Example usage
if __name__ == "__main__":
    start = time.time()

    main()
    #test()

    elapsed_time = time.time() - start
    print(f"Execution time: {elapsed_time:.3f} seconds")
    filename = f"{NAME} {datetime.now().strftime('%Y-%m-%d %H:%M')} ({elapsed_time:.3f}s)"

    if IS_REGISTERED:
        np.savez_compressed("data/" + filename,
                            hm_states=np.array(HM_STATES, dtype=object),
                            vhl_mask=VHL_MASK,
                            time=elapsed_time,
                            param=PARAMS)
