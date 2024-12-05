PARAMS = dict()
PARAMS["field_edge_length"] = 1000 # 1 meter
PARAMS["cell_edge_length"] = 5 # mm
PARAMS["grid_size"] = (PARAMS["field_edge_length"]//PARAMS["cell_edge_length"],PARAMS["field_edge_length"]//PARAMS["cell_edge_length"]) # (Ymax, Xmax)
PARAMS["k"] = PARAMS["cell_edge_length"] / 8
PARAMS["object_depth"] = 30 # mm
PARAMS["angle_of_respose"] = 30 # deg 
PARAMS["erosion_threshold"] = 10
