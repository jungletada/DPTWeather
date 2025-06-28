import os
from typing import Union, List

from .base_depth_dataset import (
    BaseDepthDataset,
    get_pred_name,  # noqa: F401
    DatasetMode,
)  # noqa: F401

from .weatherkitti_dataset import \
    WeatherKITTIDepthMixedDataset, WeatherKITTIDepthPairedDataset, WeatherKITTIDepthDataset

dataset_name_class_dict = {
    "weather_depth_kitti": WeatherKITTIDepthMixedDataset,
        # WeathewrKITTIDepthMixedDataset,
}


def get_dataset(
    cfg_data_split, 
    base_data_dir: str,
    mode: DatasetMode, 
    join_split=True, 
    **kwargs
) -> Union[
    BaseDepthDataset,
    List[BaseDepthDataset],
]:
    if "mixed" == cfg_data_split.name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset_dir = os.path.join(base_data_dir, cfg_data_split.dir) if join_split else base_data_dir
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=dataset_dir,
            **cfg_data_split,
            **kwargs,
        )
    
    else:
        raise NotImplementedError

    return dataset
