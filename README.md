# STL/G-code Noise Injection and Evaluation

This repository demonstrates a workflow for adding layer-based geometric noise to an STL, removing the corresponding layers from sliced G-code, and comparing the modified or restored toolpath with ground-truth G-code.

## Setup

Use Python 3.9 or newer. From the repository root, install the required packages:

```bash
python -m pip install numpy trimesh shapely matplotlib mapbox-earcut
```

Run all commands from the repository root because the scripts use relative folder paths.

## Workflow

1. Add noisy layers to an STL. Pass the filename without the `.stl` extension:

   ```bash
   python adding_noise_to_STL.py --file dog_bone --layer_height 0.2 --max_shift 0.2
   ```

   The input is read from `input_STLs/`, the modified STL is written to `S2_stl/`, and the inserted layer numbers are printed and saved in `noise_processing.log`.

2. Slice the modified STL with PrusaSlicer or SuperSlicer and place the generated file in `G2_gcode/`. The restoration script expects layer comments such as `;LAYER_CHANGE` and `;HEIGHT:`.

3. Open `removing_noise_G2_to_G1.py` and update:

   - `input_gcode_file`
   - `output_gcode_file`
   - `noise_array_from_stl_step` using the array produced in step 1

   Then run:

   ```bash
   python removing_noise_G2_to_G1.py
   ```

4. Compare ground-truth G-code with noisy or restored G-code:

   ```bash
   python evaluation_code.py GT_gcode/dog_bone_PLA_ENDER3.gcode G1_gcode/G1_dog_bone_LR_PLA_ENDER3.gcode --case-name dog_bone_restored
   ```

   Reports are saved under `eval_2/`. For additional plots and layer-wise statistics, run the alternative evaluator:

   ```bash
   python AML_evaluate_gcode.py GT_gcode/dog_bone_PLA_ENDER3.gcode G1_gcode/G1_dog_bone_LR_PLA_ENDER3.gcode
   ```

## Included data

- `input_STLs/`: original STL models
- `S2_stl/`: noisy STL models
- `GT_gcode/`: ground-truth G-code
- `G2_gcode/`: G-code sliced from noisy models
- `G1_gcode/`: restored G-code

