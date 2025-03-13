import sys
import os
sys.path.append(os.path.abspath(".."))  


from utils import point_clouds_to_heightmap, ros_odom_to_trajectory
import numpy as np
import open3d as o3d
import sys


if __name__ == "__main__":
    args = sys.argv
    args.pop(0)
    pcds_names = args
    traj_name = "traj.npz"
    
    cell_size = 0.02
    nb_checkpoints = 25

    pcds = [o3d.io.read_point_cloud(pcd_name) for pcd_name in pcds_names]
    heightmaps, shape = point_clouds_to_heightmap(pcds, cell_size)
    
    import matplotlib.pyplot as plt
    im_list = []
    nb_maps = len(heightmaps)
    cols = min(3, nb_maps)
    rows = (nb_maps + cols - 1) // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if nb_maps == 1:
        axes = [axes]  # Make it iterable if only 1 heightmap
    axes = np.array(axes).flatten()  # Flatten in case of multiple rows/cols

    vmin = min(np.nanmin(hmap) for hmap in heightmaps)
    vmax = max(np.nanmax(hmap) for hmap in heightmaps)
    vmax = max(abs(vmin), abs(vmax))  # Rendre la colorbar symétrique autour de 0
    vmin = -vmax

    for i, (ax, hmap, name) in enumerate(zip(axes, heightmaps,pcds_names)):
        im = ax.imshow(hmap, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        im_list.append(im)
        ax.set_title(f"{name}")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    #plt.tight_layout()
    cbar = fig.colorbar(im_list[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Height (m)")
    plt.show()
    #ros_odom_array = np.load(traj_name, allow_pickle=True)
    #trajectory = ros_odom_to_trajectory(ros_odom_array, shape, nb_checkpoints)

    np.savez_compressed(f"soil_model_data.npz",
                        heightmaps=np.array(heightmaps, dtype=object),
                        #trajectory=trajectory
                        )

