# Knowledge summary

## Overview

This file summarizes the main conclusions from the CT metadata and beam/ray geometry exploration.

The goal was to understand how the processed CT volumes, Excel metadata, and JSON beam geometry relate to each other.


## Pipeline.

Data is stored in scratch and cached into tensor checkpoints. Tensor checkpoints are tensors that can be load in extremely fast, so the training is much faster.

Data is processed by the initial transformations: Scaling, Fix Spacing, creating gantry angle prior.

Loaded data passed into the training instances.

Training instance is build from 2 components:
- **Model**: This is the architecture, the neural network that is called and shoud predict y_pred, the dosage distribution in 3d.
- **Trainer**: In the trainer we define what is steps we take to train the model, we can chose if we want adversarial training or not, or we can define extra trainer instances.

Training instance is given to pytorch lightning trainer and the loop starts.

Every important parameter is stored in the configs/userNaem_config.yaml file. You should only edit the config corresponding to your username.
 
## Data location

Training data:

    /project/nr_dose_scratch/training

Derived outputs:

    /project/nr_dose/data

Each case has an ID folder, for example:

    1ABB006
    1ABB011
    1THB226

Typical files for one case:

    /project/nr_dose_scratch/training/<ID>/image/ct.mha
    /project/nr_dose_scratch/training/<ID>/<ID>.json

Additional metadata file:

    /project/nr_dose_scratch/training/SynthRAD2025_image_parameters.xlsx

## CT metadata

The `.mha` files contain the actual processed CT volumes used by the model.

When loading a CT with MONAI, the most important metadata fields are:

- `spatial_shape`
- `spacing`
- `affine`

The `spatial_shape` field gives the processed voxel-grid size:

    spatial_shape = (Nx, Ny, Nz)

where:

- `Nx` and `Ny` are the in-plane voxel dimensions
- `Nz` is the number of voxels / slices in the z direction

The processed `.mha` files typically have spacing:

    spacing = [1, 1, 3]

So the z-spacing is usually:

    spacing_z = 3 mm

The approximate physical z-size of a CT volume is:

    Nz * spacing_z

The Excel file contains DICOM-like acquisition metadata, such as:

- `Rows`
- `Columns`
- `PixelSpacing`
- `SliceThickness`
- `Slices`

These Excel values may differ from the processed `.mha` geometry. For model input geometry, the `.mha` metadata should be treated as the relevant source.

## image_info_dict

The `image_info_dict` was created to collect useful metadata for each ID.

Example structure:

    image_info_dict["1ABB006"] = {
        "spatial_shape": [498, 493, 164],
        "PixelSpacing_excel": [0.625, 0.625]
    }

Meaning:

- `spatial_shape`: actual processed `.mha` voxel shape
- `PixelSpacing_excel`: pixel spacing value read from the Excel metadata file

This dictionary is useful for comparing the final processed CT geometry with the original Excel metadata.

## JSON beam/ray geometry

Each case has a JSON file describing the irradiation geometry.

Important fields:

- `iso_center`: physical isocenter coordinate
- `beams`: list of beam directions
- `beam_idx`: beam index
- `gantry_angle`: gantry angle in degrees
- `rays`: list of rays within a beam
- `ray_idx`: ray index within a beam
- `ray_source`: physical source point of the ray
- `ray_target`: physical target point of the ray
- `beamlets`: energy components associated with the ray

A single ray is defined by two physical 3D points:

- `ray_source`
- `ray_target`

The ray direction is determined by the vector from `ray_source` to `ray_target`.

## df_coords

The `df_coords` DataFrame was created from all JSON files.

Each row corresponds to one ray.

Important columns:

- `ID`
- `beam_idx`
- `gantry_angle`
- `ray_idx`
- `source_x`, `source_y`, `source_z`
- `target_x`, `target_y`, `target_z`
- `diff_x`, `diff_y`, `diff_z`

The full table contains:

    40500 rows

This comes from:

    75 IDs
    36 beams per ID
    15 rays per beam

So:

    75 * 36 * 15 = 40500 rays

## Main ray-geometry finding

For every ray, the source and target z-coordinates are equal:

    source_z = target_z

This was verified using the `diff_z` column.

For all rays:

    diff_z = 0

This means that an individual ray does not move upward or downward in the z direction. Each ray lies in one fixed z-plane.

Example:

    ray_source = [-16.13, -1062.03, 33.99]
    ray_target = [-16.13,   -62.03, 33.99]

Both points have:

    z = 33.99

So this ray lies entirely in the plane:

    z = 33.99 mm

## Beam structure

Although a single ray has constant z, one beam contains multiple rays at different z-levels.

For example, for `1ABB006`, the unique ray z-levels were:

    [33.99, 53.99, 73.99, 93.99, 113.99]

So the ray grid is defined on 5 discrete z-levels.

Each beam has 15 rays. The structure can be interpreted as:

    3 lateral positions
    5 z-levels
    total: 15 rays per beam

Useful mental model:

    one ray  = one line at fixed z
    one beam = multiple rays arranged over several z-levels

The `gantry_angle` changes the beam direction around the patient. The different `ray_idx` values define the individual rays inside that beam.

## z_min and z_max

For each ID, `z_min` and `z_max` were computed from the ray z-coordinates.

Interpretation:

- `z_min` is the lowest z-level among the rays
- `z_max` is the highest z-level among the rays
- they are not the CT volume boundaries
- they describe the z-range of the ray grid

Because every ray has `source_z = target_z`, the same result is obtained whether we use source z-values or target z-values.

For all IDs, the ray-grid z-extent was:

    z_max - z_min = 80 mm

This happens because the rays are placed on 5 z-levels separated by 20 mm.

The ray z-levels are approximately centered around the isocenter:

    iso_center_z - 40
    iso_center_z - 20
    iso_center_z
    iso_center_z + 20
    iso_center_z + 40

Therefore, the full ray-grid z-range is 80 mm.

## CT z-extent vs ray-grid z-extent

The CT volume is usually much larger in the z direction than the ray grid.

The CT z-extent was approximated as:

    Nz * spacing_z

The ray-grid z-extent was computed as:

    z_max - z_min

Example for `1ABB006`:

    Nz = 164
    spacing_z = 3 mm
    CT z-extent ≈ 492 mm
    ray-grid z-extent = 80 mm
    ray-grid z-extent / CT z-extent ≈ 16.26%

This percentage only compares the vertical size of the discrete ray grid with the full CT volume size.

It does not mean that only this percentage of the CT receives dose.

## In-plane CT shape variability

The first two components of `spatial_shape` are the in-plane dimensions:

    Nx
    Ny

The in-plane voxel area was computed as:

    Nx * Ny

The variability of `Nx`, `Ny`, and `Nx * Ny` was checked across cases using median-based percentage deviations.

This helps determine how much the CT grid size varies between patients.

Large variability in these values may matter for preprocessing, because model training may require consistent input sizes using cropping, padding, resizing, or resampling.

## Main conclusions

- The `.mha` files contain the actual processed CT volumes used by the model.
- The relevant CT geometry comes from `.mha` metadata: `spatial_shape`, `spacing`, and `affine`.
- The Excel metadata is useful for comparison but may not match the final processed CT geometry.
- Each case has a JSON file containing isocenter, beam, ray, and beamlet information.
- Each ID has 36 beams.
- Each beam has 15 rays.
- The full ray-coordinate table contains 40500 rays.
- For every ray, `source_z = target_z`.
- Therefore, each individual ray lies in one fixed z-plane.
- A beam consists of multiple rays distributed across 5 discrete z-levels.
- The ray-grid z-extent is constant across cases: 80 mm.
- `z_min` and `z_max` describe the lowest and highest ray z-levels, not the CT volume boundaries.
- The CT volume is the full voxel domain, while the ray grid is a smaller geometry grid defined by the JSON beam/ray structure.
- In-plane CT dimensions vary across cases, so preprocessing choices should take this variability into account.