import re

def strip_noise_from_gcode(input_path, output_path, noise_layer_array):
    """
    Parses a G-code file, removes specific layers, and dynamically adjusts
    all subsequent Z-heights to prevent mid-air printing.
    """
    # Convert array to a set for faster lookups
    noise_set = set(noise_layer_array)
    
    with open(input_path, 'r') as file:
        lines = file.readlines()

    output_lines = []
    
    # State tracking
    current_layer = 0
    skip_current_layer = False
    z_offset = 0.0
    layer_height = 0.2  # Default fallback, updates dynamically
    
    # Regex to find Z coordinates (matches G1 Z0.4, G1 Z.4, or ;Z:0.4)
    z_pattern = re.compile(r'([Zz]:?\s*)([0-9]*\.?[0-9]+)')

    for line in lines:
        # 1. Dynamically grab the layer height from Prusa/SuperSlicer comments
        if line.startswith(';HEIGHT:'):
            try:
                layer_height = float(line.strip().split(':')[1])
            except ValueError:
                pass

        # 2. Detect layer transitions
        if line.startswith(';LAYER_CHANGE'):
            current_layer += 1
            if current_layer in noise_set:
                skip_current_layer = True
                z_offset += layer_height
                continue  # Drop the LAYER_CHANGE comment itself
            else:
                skip_current_layer = False
        
        # 3. Drop the lines if we are inside a noise layer
        if skip_current_layer:
            continue

        # 4. Apply Z-height compensation to the layers we keep
        if z_offset > 0.0:
            # We only modify movement commands and explicit Z-height comments
            if line.startswith(('G0', 'G1', ';Z:')):
                def z_replacer(match):
                    prefix = match.group(1)
                    original_z = float(match.group(2))
                    # Apply the shift, preventing the nozzle from diving into the bed
                    new_z = max(0.0, original_z - z_offset) 
                    
                    # Format cleanly, stripping trailing zeros
                    formatted_z = f"{new_z:.3f}".rstrip('0').rstrip('.') if '.' in f"{new_z:.3f}" else f"{new_z}"
                    return f"{prefix}{formatted_z}"
                    
                line = z_pattern.sub(z_replacer, line)

        output_lines.append(line)

    # Write the adjusted G-code
    with open(output_path, 'w') as file:
        file.writelines(output_lines)

    print("\n--- G-code Processing Complete ---")
    print(f"Target noise layers removed: {len(noise_layer_array)}")
    print(f"Final Z-height reduction applied: {z_offset:.2f}mm")

# ==========================================
# Example Execution:
# ==========================================
input_gcode_file = "./G2_gcode/S2_dog_bone_0.2_1h6m_0.20mm_205C_PLA_ENDER3.gcode"
output_gcode_file = "./G1_gcode/G1_dog_bone_LR_PLA_ENDER3.gcode"

# Paste the array output from your previous script here
# noise_array_from_stl_step = [6, 9, 11, 14, 16, 19, 21, 24, 26, 29, 31, 34, 36, 40, 42, 45, 47, 50, 52, 55, 58, 60, 63, 65, 68, 70, 73, 75, 78, 80, 83, 85, 88, 90, 93, 95, 98, 101, 103, 107, 109, 112]
noise_array_from_stl_step = [6, 8, 11, 13, 16]

strip_noise_from_gcode(input_gcode_file, output_gcode_file, noise_array_from_stl_step)