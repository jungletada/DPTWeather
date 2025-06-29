"""Compute depth maps for images in the input folder.
"""
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt.models import DPTDepthModel

from src.dataset import (
    DatasetMode,
    get_dataset,
    get_pred_name,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="DPT for Monocular Depth Estimation"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output/prediction",
        help="folder for output images",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default='config/dataset_depth/data_kitti_eigen_test.yaml',
        help="Path to the config file of the evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default='data/kitti',
        help="Base path to the datasets.",
    )
    parser.add_argument(
        "-m", "--model_weights", 
        default='weights/dpt_hybrid_kitti-cb926ef4.pt', 
        help="path to model weights"
    )
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid_kitti",
        help="model type [dpt_large|dpt_hybrid|dpt_hybrid_kitti]",
    )
    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)

    return parser.parse_args()
  
    
def run(args, dataset):
    """
        Run MonoDepthNN to compute depth maps.
    """
    print("Initialize")
    model_path = args.model_weights
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load network
    if args.model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    elif args.model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    elif args.model_type == "dpt_hybrid_kitti":
        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    else:
        assert (
            False
        ), f"model_type '{args.model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti]"

    model.eval()

    if args.optimize and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)

    model.to(device)
    
    print("Start processing")
    for ind, sample_data in enumerate(tqdm(dataset, desc="Processing images")):
        img_input = sample_data['rgb_norm']
        rel_path = sample_data['rgb_relative_path']

        with torch.no_grad():
            sample = img_input.to(device).unsqueeze(0)
            if args.optimize and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=sample.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy())
            
            # Save prediction as npy file
            npy_filename = rel_path.replace('.png', '.npy')
            npy_path = os.path.join(args.output_path, npy_filename)
            
            # Create all necessary subdirectories
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            
            np.save(npy_path, prediction.astype(np.float32))


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }
    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # -------------------- Data -----------------------
    cfg_data = OmegaConf.load(args.dataset_config)
    dataset = get_dataset(
        cfg_data, 
        base_data_dir=args.base_data_dir, 
        mode=DatasetMode.EVAL,
        join_split=False,
    )
    # compute depth maps
    run(args, dataset)
