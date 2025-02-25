# Synth2 Dataset Structure

## Overview
The Synth2 dataset is organized into a structured format containing RGB images and their corresponding depth maps. The dataset is split into two main directories: one for RGB images and another for depth maps.

## Directory Structure
```
synth2_dataset/
├── synth2_rgb/
│   └── [scene_id]/
│       └── images/
│           ├── 001.jpeg
│           ├── 002.jpeg
│           ├── 003.jpeg
│           ├── ...
│           ├── 024.jpeg
└── synth2_depth/
    └── [scene_id]/
        └── depth_maps/
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
            ├── 024.png
```

Where:
- `[scene_id]`: Unique identifier for each scene (e.g., `3c0cb41238af4330b81d1340cc81ff94`)

## Content Organization

### 1. RGB Images (synth2_rgb/):
   - Contains original RGB images
   - Located in `images/` subdirectory within each scene
   - Format: JPEG
   - Naming convention: `001.jpeg`, `002.jpeg`, ..., `024.jpeg`

### 2. Depth Maps (synth2_depth/):
   - Contains corresponding depth map images
   - Located in `depth_maps/` subdirectory within each scene
   - Format: PNG
   - Naming convention: `001.png`, `002.png`, ..., `024.png`

## Dataset Statistics
- **Total number of scenes**: 100
- **Each scene contains multiple sequential frames**
- **One-to-one correspondence between RGB and depth images**

## File Correspondence
For each RGB image in `synth2_rgb/[scene_id]/images/`, there is a corresponding depth map in `synth2_depth/[scene_id]/depth_maps/` with the same filename but different extensions.
