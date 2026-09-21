import os

import trimesh
import numpy as np
import random
from shapely.geometry import Polygon
from shapely.affinity import translate
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('noise_processing.log'),
        logging.StreamHandler()
    ]
)

def generate_insertion_sequence(num_eligible):
    """Generates the 1/0 sequence based on the 2/3 target and spacing rules."""
    sequence = []
    for i in range(num_eligible):
        if len(sequence) >= 2 and sequence[-1] == 1 and sequence[-2] == 1:
            sequence.append(0)
        else:
            sequence.append(1 if random.random() < 0.85 else 0)
    return sequence

def apply_layer_shift(polygon, max_shift=0.2):
    """
    Randomly shifts the polygon left (-X), right (+X), or not at all (0).
    max_shift is kept small (e.g., 0.2mm) to ensure perimeter overlap and prevent gaps.
    """
    if polygon is None or polygon.is_empty:
        return polygon

    # Choice: -1 (Left), 0 (No Shift), 1 (Right)
    shift_choice = random.choice([-1, 0, 1])
    
    if shift_choice == 0:
        return polygon
        
    # Translate the polygon along the X-axis
    shift_amount = shift_choice * max_shift
    return translate(polygon, xoff=shift_amount, yoff=0.0)

def process_and_reconstruct_stl(file_path, output_path, layer_height=0.2, max_shift=0.2):
    print(f"Loading mesh from {file_path}...")
    mesh = trimesh.load(file_path)
    
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    z_levels = np.arange(z_min + (layer_height / 2), z_max, layer_height)
    total_original_layers = len(z_levels)
    
    if total_original_layers <= 8:
        raise ValueError("Mesh is too small to exclude first/last 4 layers.")

    # Determine insertion logic
    insertion_mask = [0] * total_original_layers
    eligible_indices = list(range(4, total_original_layers - 4))
    
    sequence = generate_insertion_sequence(len(eligible_indices))
    for idx, val in zip(eligible_indices, sequence):
        insertion_mask[idx] = val

    all_extruded_meshes = []
    noise_layer_array = [] # Array tracking the final indices of inserted layers
    
    current_z_height = z_min
    final_layer_index = 1 
    
    for i, z_original in enumerate(z_levels):
        slice_3d = mesh.section(plane_origin=[0, 0, z_original], plane_normal=[0, 0, 1])
        if slice_3d is None:
            continue
            
        slice_2d, to_3d_transform = slice_3d.to_planar()
        polygons = slice_2d.polygons_full
        
        # --- 1. PROCESS ORIGINAL LAYER ---
        for poly in polygons:
            try:
                ex_mesh = trimesh.creation.extrude_polygon(poly, height=layer_height)
                ex_mesh.apply_transform(to_3d_transform)
                z_shift = current_z_height - z_original
                ex_mesh.apply_translation([0, 0, z_shift])
                all_extruded_meshes.append(ex_mesh)
            except Exception:
                pass 
                
        current_z_height += layer_height
        final_layer_index += 1
        
        # --- 2. PROCESS SHIFTED NOISE LAYER ---
        if insertion_mask[i] == 1:
            noise_layer_array.append(final_layer_index)
            
            for poly in polygons:
                # Apply the randomized left/right/center shift
                shifted_poly = apply_layer_shift(poly, max_shift=max_shift)
                
                try:
                    ex_mesh = trimesh.creation.extrude_polygon(shifted_poly, height=layer_height)
                    ex_mesh.apply_transform(to_3d_transform)
                    z_shift = current_z_height - z_original
                    ex_mesh.apply_translation([0, 0, z_shift])
                    all_extruded_meshes.append(ex_mesh)
                except Exception:
                    pass
                    
            current_z_height += layer_height
            final_layer_index += 1

    print("Concatenating geometry...")
    final_mesh = trimesh.util.concatenate(all_extruded_meshes)
    final_mesh.export(output_path)
    
    print("\n--- Process Complete ---")
    print(f"Total noise layers added: {len(noise_layer_array)}")
    
    # Returning the requested array
    return noise_layer_array

# ==========================================
# Example Execution:
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--layer_height", type=float, required=True)
parser.add_argument("--max_shift", type=float, required=True)
parser.add_argument("--file", type=str, required=True)
args = parser.parse_args()

layer_height_var = args.layer_height
shift_distance_var = args.max_shift
input_stl_name = args.file
# input_stl = "./random_mix_idea/input_STLs/" + input_stl_name + ".stl"
input_stl = "./input_STLs/" + input_stl_name + ".stl"
output_stl = "./S2_stl/Auto_modifiedZ_" + input_stl_name + "_" + str(layer_height_var) + ".stl"

noise_array = process_and_reconstruct_stl(
    input_stl, 
    output_stl, 
    layer_height=layer_height_var, 
    max_shift=shift_distance_var
)
logging.info("Array of Noise Layers: %s", noise_array)
logging.info("File Name: %s", os.path.basename(input_stl))
logging.info("layer_height: %s", layer_height_var)
logging.info("Output STL saved to: %s", output_stl)
logging.info("-------------------------------")
