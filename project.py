import numpy as np
import time
from datetime import datetime
import tqdm
from utils import *
from config import *
from sand_simulator import SandSimulator
from vehicule import Vehicule

np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200, threshold=np.inf)

HM_STATES = list()
VHL_MASK = None  
NAME = None

#TODO better display of tqdm, fix unit/s

def run():
    """
    Main function to initialize the sand simulator and simulate vehicule movement
    along a defined trajectory.
    """
    global HM_STATES, VHL_MASK, NAME
    NAME = "traj crossed"

    # Initialize the simulator
    simulator = SandSimulator(grid_size=PARAMS["grid_size"])
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))

    # Create a vehicule (e.g., a bulldozer blade)
    VHL_MASK = ellipse_generator(30 // PARAMS["cell_edge_length"], 50 // PARAMS["cell_edge_length"])
    vhl = Vehicule(mask=VHL_MASK, simulator=simulator)

    # Define a trajectory for the vehicule
    traj_comp = list()
    traj_comp.append(generate_linear_trajectory((100, 100), (800, 800), 20))
    traj_comp.append(generate_linear_trajectory((800, 100), (100, 800), 20))
    #traj_comp.append(generate_sinusoidal_trajectory((100,100), (700,700), 2, 100, 25))
    traj = np.concatenate(traj_comp) // PARAMS["cell_edge_length"] 
    # Make the vehicule follow the trajectory
    vhl.follow_trajectory(traj)

    #Last clean HM_state
    vhl.position = None 
    simulator.simulate_erosion()
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.vehicule_trajectory)))

def speed_measure():

    pool = dict()
    cell_len_values = [100,50,40,25,20,10,8,5,4,2,1]

    for i in cell_len_values:
        PARAMS["cell_edge_length"] = i
        conf_update()
        start = time.time()
        run()
        elapsed_time = time.time() - start

        pool[i]= f"{elapsed_time:.3f}"
        print(pool)

    np.savez("data/speedups/notspeeded cell_len_values.npy",pool=pool, param=PARAMS,allow_pickle=True)



def register_run():
    start = time.time()

    run()

    elapsed_time = time.time() - start

    print(f"Execution time: {elapsed_time:.3f} seconds")
    filename = f"{NAME} {datetime.now().strftime('%Y-%m-%d %H:%M')} ({elapsed_time:.3f}s)"

    np.savez_compressed("data/runs/" + filename,
                        hm_states=np.array(HM_STATES, dtype=object),
                        vhl_mask=VHL_MASK,
                        time=elapsed_time,
                        param=PARAMS)



# Example usage
if __name__ == "__main__":
    register_run()
    #speed_measure()
