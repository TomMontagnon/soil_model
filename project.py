"""
Module: Sand Simulation and Vehicle Trajectory Runner
Simulates vehicle movement on sand-like surfaces, generates trajectories, and benchmarks performance.
"""

import numpy as np
import time
from datetime import datetime
import tqdm
from utils import *
from config import *
from sand_simulator import SandSimulator
from vehicule import Vehicule


# Set numpy display options for clarity
np.set_printoptions(precision=0)
np.set_printoptions(linewidth=200, threshold=np.inf)



def traj_generator(traj_type):
    """Generate vehicle trajectories of type 'X' or 'S' with checkpoints."""

    if PARAMS["nb_checkpoints"] < 2:
        raise ValueError("At least two checkpoints required.")

    traj_comp = list()
    if traj_type == "X":
        traj_comp.append(generate_linear_trajectory((100, 100), (800, 800), PARAMS["nb_checkpoints"] // 2))
        traj_comp.append(generate_linear_trajectory((800, 100), (100, 800), PARAMS["nb_checkpoints"] // 2))
    elif traj_type == "S":
        traj_comp.append(generate_sinusoidal_trajectory((100,100), (800,800), 2, 100, PARAMS["nb_checkpoints"]))
    else:
        raise ValueError(f"Unsupported trajectory type: {traj_type}")

    # Merge and scale trajectory parts
    traj = np.concatenate(traj_comp) // PARAMS["cell_edge_length"] 
    return traj


def run(traj_type):
    """Run a simulation where a vehicle follows a given trajectory."""

    global VHL_MASK, TRAJ_PLANNED

    # Initialize the simulator
    simulator = SandSimulator(grid_size=PARAMS["grid_size"])

    # Create a vehicule
    VHL_MASK = ellipse_generator(PARAMS["ellipse_semi_minor_axis"], PARAMS["ellipse_semi_major_axis"])
    vhl = Vehicule(mask=VHL_MASK, simulator=simulator)

    # Make the vehicule follow the trajectory
    TRAJ_PLANNED = traj_generator(traj_type)
    vhl.follow_trajectory(TRAJ_PLANNED)

    #Perform one last erosion withouth the vehicule
    simulator.final_erosion()





def register_speedup():
    """Measure simulation time across multiple resolutions."""

    name = "cell_edge_length"
    unit = "[mm]"
    git_tag = "v2.0"
    traj_type = "X"
    measures = dict()
    hm_states = dict()
    parameter_variations = [100,50,40,25,20,10,8,5,4,2]
    #parameter_variations = [100,80,60,40,30,25,20,15,10]

    run(traj_type) #Perform fake run to compile NUMBA function

    for param in parameter_variations:
        PARAMS[name] = param
        conf_update()
        start = time.time()
        run(traj_type)
        elapsed_time = time.time() - start

        measures[param] = f"{elapsed_time:.3f}"
        hm_states[param] = HM_STATES[-1][0]
        print(measures)


    filename = f"{name} {git_tag} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    np.savez_compressed(f"data/speedups/{filename}",
                        git_tag=git_tag,
                        traj_type=traj_type,
                        traj_planned=TRAJ_PLANNED,
                        param=PARAMS,
                        name=name,
                        measures=measures,
                        unit=unit,
                        hm_states=hm_states)

def register_run():
    """Run a single simulation and save time and parameters."""

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
    """Simulate sand erosion on a small grid and save results."""

    name = "erosion"
    PARAMS["grid_size"] = (25,25)
    height_map_center = tuple(np.array(PARAMS["grid_size"]) // 2)
    
    # Initialize the simulator
    simulator = SandSimulator(grid_size=PARAMS["grid_size"])
    simulator.height_map[height_map_center] = 1000

    # Create a vehicule
    VHL_MASK = ellipse_generator(PARAMS["ellipse_semi_minor_axis"], PARAMS["ellipse_semi_major_axis"])
    vhl = Vehicule(mask=VHL_MASK, simulator=simulator)

    # Make one erosion process
    simulator.simulate_erosion(register=True)

    filename = f"{name} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    np.savez_compressed(f"data/erosions/{filename}",
                        hm_states=np.array(HM_STATES, dtype=object),
                        param=PARAMS)

# Example usage (comment/uncomment as needed)
if __name__ == "__main__":
    #register_run()
    register_speedup()
    #register_erosion()
