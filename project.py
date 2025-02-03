import numpy as np
import time
from datetime import datetime
import tqdm
from utils import *
from config import *
from sand_simulator import SandSimulator
from vehicule import Vehicule

np.set_printoptions(precision=0)
np.set_printoptions(linewidth=200, threshold=np.inf)



def traj_generator(traj_type):
    traj_comp = list()

    if traj_type == "X":
        traj_comp.append(generate_linear_trajectory((100, 100), (800, 800), PARAMS["nb_checkpoints"] // 2))
        traj_comp.append(generate_linear_trajectory((800, 100), (100, 800), PARAMS["nb_checkpoints"] // 2))
    elif traj_type == "S":
        traj_comp.append(generate_sinusoidal_trajectory((100,100), (800,800), 2, 100, PARAMS["nb_checkpoints"]))
    else:
        raise ValueError(f"Unknown type_traj : {type_traj}")
    traj = np.concatenate(traj_comp) // PARAMS["cell_edge_length"] 
    return traj


def run(traj_type):
    """
    Main function to initialize the sand simulator and simulate vehicule movement
    along a defined trajectory.
    """
    global VHL_MASK, TRAJ_PLANNED
    # Initialize the simulator
    simulator = SandSimulator(grid_size=PARAMS["grid_size"])

    # Create a vehicule (e.g., a bulldozer blade)
    VHL_MASK = ellipse_generator(PARAMS["ellipse_semi_minor_axis"], PARAMS["ellipse_semi_major_axis"])
    vhl = Vehicule(mask=VHL_MASK, simulator=simulator)

    # Make the vehicule follow the trajectory
    TRAJ_PLANNED = traj_generator(traj_type)
    vhl.follow_trajectory(TRAJ_PLANNED)

    #Last clean HM_state
    simulator.final_erosion()





def register_run_speedup():
    name = "cell_len_values"
    git_tag = "v1.0"
    traj_type = "X"
    measures = dict()

    parameter_variations = [100,50,40,25,20,10,8,5,4,2,1]
    parameter_variations = [100,50,40,25,20,10,8,5,4]

    for param in parameter_variations:
        PARAMS["cell_edge_length"] = param
        conf_update()

        start = time.time()
        run(traj_type)
        elapsed_time = time.time() - start

        measures[param]= f"{elapsed_time:.3f}"
        print(measures)


    filename = f"{name} {git_tag} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    np.savez_compressed(f"data/speedups/{filename}",
                        git_tag=git_tag,
                        traj_type=traj_type,
                        traj_planned=TRAJ_PLANNED,
                        param=PARAMS,
                        name=name,
                        measures=measures)

def register_run():
    traj_type = "X"
    name = f"traj {traj_type}"

    start = time.time()
    run(traj_type)
    elapsed_time = time.time() - start

    filename = f"{name} {datetime.now().strftime('%Y-%m-%d %H:%M')} ({elapsed_time:.3f}s)"

    np.savez_compressed(f"data/runs/{filename}",
                        hm_states=np.array(HM_STATES, dtype=object),
                        vhl_mask=VHL_MASK,
                        time=elapsed_time,
                        param=PARAMS)

def register_erosion():
    name = "erosion"

    # Initialize the simulator
    simulator = SandSimulator(grid_size=PARAMS["grid_size"])
    height_map_center = tuple(np.array(PARAMS["grid_size"]) // 2)
    simulator.height_map[height_map_center] = 10000
    simulator.simulate_erosion_cuda(register=True)


    filename = f"{name} {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    np.savez_compressed(f"data/erosions/{filename}",
                        hm_states=np.array(HM_STATES, dtype=object),
                        param=PARAMS)

# Example usage
if __name__ == "__main__":
    #register_run()
    register_run_speedup()
    #register_erosion()
