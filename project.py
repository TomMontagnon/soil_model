import numpy as np
import time
from datetime import datetime

from utils import *
from config import PARAMS
from sand_simulator import SandSimulator
from vehicule import Object

#TODO faire 2 affichages sur jupyter
#   - 1 compliqué qui fait un petit film (cf chatgpt)
#TODO mettre self.position par defaut à None et réparer
#TODO tout rename par vehicule



IS_REGISTERED = True
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200, threshold=np.inf)

HM_STATES = list()
TRAJ_STATES = list()
OBJ_MASK = None
NAME = None

def main():
    global HM_STATES, OBJ_MASK, NAME
    NAME = "traj X"

    # Initialize the simulator
    simulator = SandSimulator()
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.object_trajectory)))

    # Create a object (e.g., a bulldozer blade)
    OBJ_MASK= ellipse_generator(30 // PARAMS["cell_edge_length"] ,50 //PARAMS["cell_edge_length"])
    obj = Object(mask=OBJ_MASK, simulator=simulator)

    # Define a trajectory for the object
    traj1 = generate_linear_trajectory((100, 100), (900, 900), 3) // PARAMS["cell_edge_length"]- np.array(obj.mask_center)
    traj2 = generate_linear_trajectory((100, 900), (900, 100), 1) // PARAMS["cell_edge_length"]- np.array(obj.mask_center)
    traj = np.concatenate((traj1,traj2))
    print(traj)
    # Make the object follow the trajectory
    obj.follow_trajectory(traj)
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.object_trajectory)))

def test():
    global HM_STATES, OBJ_MASK, NAME

    NAME = "test"

    simulator = SandSimulator(grid_size = (20,20))
    simulator.height_map[10,10] = 1000
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.object_trajectory)))
    
    OBJ_MASK = ellipse_generator(4,4)
    obj = Object(mask=OBJ_MASK, simulator=simulator)
    
    simulator.simulate_erosion()
    HM_STATES.append((simulator.height_map.copy(), np.array(simulator.object_trajectory)))


# Example usage
if __name__ == "__main__":
    start = time.time()

    main()
    #test()

    time = time.time() - start
    print(f"Temps d'exécution : {time:.3f} secondes")
    filename = f"{NAME} {datetime.now().strftime("%Y-%m-%d %H:%M")} ({time:.3f}s)"
    if IS_REGISTERED:
        np.savez_compressed("data/"+filename,
                        hm_states=np.array(HM_STATES, dtype=object),
                        obj_mask=OBJ_MASK,
                        time=time,
                        param=PARAMS)

