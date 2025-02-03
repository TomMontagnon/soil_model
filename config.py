PARAMS = dict()
PARAMS["field_edge_length"] = 1000 # 1 meter
PARAMS["cell_edge_length"] = 5 # mm
PARAMS["grid_size"] = (PARAMS["field_edge_length"]//PARAMS["cell_edge_length"],PARAMS["field_edge_length"]//PARAMS["cell_edge_length"]) # (Ymax, Xmax)
PARAMS["k"] = 1 / 8
PARAMS["vehicule_depth"] = 30 # mm
PARAMS["angle_of_respose"] = 30 # deg 
PARAMS["erosion_threshold"] = 10
PARAMS["nb_checkpoints"] = 20
PARAMS["ellipse_semi_major_axis"] = 50 // PARAMS["cell_edge_length"]
PARAMS["ellipse_semi_minor_axis"] = 30 // PARAMS["cell_edge_length"]

def conf_update():
    PARAMS["grid_size"] = (PARAMS["field_edge_length"]//PARAMS["cell_edge_length"],PARAMS["field_edge_length"]//PARAMS["cell_edge_length"]) # (Ymax, Xmax)
    PARAMS["ellipse_semi_major_axis"] = 50 // PARAMS["cell_edge_length"]
    PARAMS["ellipse_semi_minor_axis"] = 30 // PARAMS["cell_edge_length"]


#GLOBALS
HM_STATES = list()
TRAJ_PLANNED = None
VHL_MASK = None  

