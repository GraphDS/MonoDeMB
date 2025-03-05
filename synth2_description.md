# Synth2 Dataset Structure

## Overview
The Synth2 dataset is organized into a structured format containing RGB images and their corresponding depth maps. The dataset is split into two main directories: one for RGB images and another for depth maps.

## Directory Structure
```
synth2_dataset/
├── [UUID]/                 # Scene ID (e.g., 016aa8df34e946faa02a145bdabe4b48)
│   ├── rgb/                # RGB images directory
│   │   ├── 1.jpeg
│   │   ├── 2.jpeg
│   │   ├── ...
│   │   └── 24.jpeg
│   └── gt/            # Ground truth depth maps
│       ├── 1.png
│       ├── 2.png
│       ├── ...
│       └── 24.png
├── [UUID2]/                # Another scene
│   ├── rgb/
│   └── gt/
└── ...
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
- **Total number of scenes**: 300 (metrics gained on 100)
- **Each scene contains multiple sequential frames**
- **One-to-one correspondence between RGB and depth images**

## Versions and Download

We present 2 versions of our dataset Synth2: [small](https://ue-benchmark-dp.obs.ru-moscow-1.hc.sbercloud.ru/synth2.tar.gz) (100 samples, the results showed in the paper) and [full](https://ue-benchmark-dp.obs.ru-moscow-1.hc.sbercloud.ru/synth2_v2.tar.gz) (300 samples)

## File Correspondence
For each RGB image in `synth2_rgb/[scene_id]/images/`, there is a corresponding depth map in `synth2_depth/[scene_id]/depth_maps/` with the same filename but different extensions.
