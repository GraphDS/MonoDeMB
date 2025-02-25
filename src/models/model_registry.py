MODEL_REGISTRY = {
    "unidepth": {
        "module": "src.models.unidepth_wrapper",
        "class": "UniDepthModelWrapper",
        "variants": [
            "unidepth-v2-vitl14",
            "unidepth-v2-vitb14", 
            "unidepth-v1-vitb14",
            "unidepth-v1-dpt-hybrid",
        ]
    },
    "depth_anything": {
        "module": "src.models.depth_anything_wrapper",
        "class": "DepthAnythingWrapper",
        "variants": ["vitl", "vitb", "vits"]
    },
    "midas": {
        "module": "src.models.midas",
        "class": "MidasModel",
        "variants": ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    },
    "marigold": {
        "module": "src.models.marigold_wrapper",
        "class": "MarigoldWrapper",
        "variants": ["Original", "LCM"]
    },
    "metric3dv2": {
        "module": "src.models.metric3d_v2_wrapper",
        "class": "Metric3DV2Wrapper",
        "variants": ["vit_small", "vit_base", "vit_large"]
    },
    "leres": {
        "module": "src.models.leres_wrapper",
        "class": "LeReSWrapper",
        "variants": ["resnext101", "resnet50"]
    },
    "genpercept": {
        "module": "src.models.genpercept_wrapper",
        "class": "GenPerceptWrapper",
        "variants": ["sd21", "sd21-lora", "sd21-rgb-blending"]
    },
    "geowizard": {
        "module": "src.models.geowizard_wrapper",
        "class": "GeoWizardWrapper", 
        "variants": ["v1", "v2"]
    }
}

# For backward compatibility
MODEL_VARIANTS = {name: info["variants"] for name, info in MODEL_REGISTRY.items()}