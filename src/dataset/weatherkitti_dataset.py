# Modified from Marigold:
import os
import torch
import random
import torch.nn.functional as F
from .base_depth_dataset import \
    BaseDepthDataset, DepthFileNameMode, DatasetMode


class WeatherOption:
    def __init__(self):
        self.fog_path_1 = ('fog', '75m')
        self.fog_path_2 = ('fog', '150m')
        self.rain_path = ('mix_rain', '50mm')
        self.snow_path = ('mix_snow', 'data')
        self.raingan_path = ('raingan', 'data')
        self.snowgan_path = ('snowgan', 'data')
        self.num_domains = 6
        

class WeatherKITTIDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        kitti_bm_crop=True,  # Crop to KITTI benchmark size
        train_crop=None,
        squeeze_channel=False,
        valid_mask_crop='eigen',  # Evaluation mask. [None, garg or eigen]
        **kwargs,
    ) -> None:
        super().__init__(
            # KITTI data parameter
            min_depth=1e-6,
            max_depth=80,
            has_filled_depth=True,
            name_mode=DepthFileNameMode.id,
            **kwargs)
        
        self.weather_opt = WeatherOption()
        self.rgb_path = 'rgb'
        self.depth_path = 'depth'
        self.filled_depth_path = 'align_depth'
        
        self.KB_CROP_HEIGHT = 352
        self.KB_CROP_WIDTH = 1216
        self.train_crop = train_crop
        self.squeeze_channel = squeeze_channel
        if self.train_crop is not None and self.mode != 'train':
            print("Error! You can not set train_crop while not training.")
            exit(0)

        self.kitti_bm_crop = kitti_bm_crop
        self.valid_mask_crop = valid_mask_crop
        assert self.valid_mask_crop in [
            None,
            "garg",   # set evaluation mask according to Garg ECCV16
            "eigen",  # set evaluation mask according to Eigen NIPS14
        ], f"Unknown crop type: {self.valid_mask_crop}"
        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]
        
        self.rgb_files = [os.path.join(self.rgb_path, filename_line[0][11:])
                          for filename_line in self.filenames]
        self.depth_files = [os.path.join(self.depth_path, filename_line[1])
                          for filename_line in self.filenames]
        
        self.aligned_depth_files = [os.path.join(self.filled_depth_path, filename_line[0][11:])
                          for filename_line in self.filenames]
        
        self.filled = [filename_line[2] for filename_line in self.filenames]
        
        self.rain_files = [os.path.join(self.weather_opt.rain_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.rain_path[1]))
                            for filename_line in self.filenames]
        self.raingan_files = [os.path.join(self.weather_opt.raingan_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.raingan_path[1]))
                            for filename_line in self.filenames]
        self.fog1_files = [os.path.join(self.weather_opt.fog_path_1[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_1[1]))
                            for filename_line in self.filenames]
        self.fog2_files = [os.path.join(self.weather_opt.fog_path_2[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.fog_path_2[1]))
                            for filename_line in self.filenames]
        self.snow_files = [os.path.join(self.weather_opt.snow_path[0], 
                                        filename_line[0][11:].replace('data', self.weather_opt.snow_path[1]))
                          for filename_line in self.filenames]
        self.snowgan_files = [os.path.join(self.weather_opt.snowgan_path[0], 
                                           filename_line[0][11:].replace('data', self.weather_opt.snowgan_path[1]))
                          for filename_line in self.filenames]

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode KITTI depth
        depth_decoded = depth_in / 256.0
        return depth_decoded
    
    def _load_rgb_data(self, rgb_rel_path, dict_names=("rgb_int", "rgb_norm")):
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_data = {
            dict_names[0]: torch.from_numpy(rgb).int(),
            dict_names[1]: torch.from_numpy(rgb_norm).float(),
        }
        if self.kitti_bm_crop:
            rgb_data = {k: self.kitti_benchmark_crop(v) for k, v in rgb_data.items()}
            
        if self.train_crop is not None:
            rgb_data = {k: self.kitti_train_crop(v) for k, v in rgb_data.items()}
        return rgb_data
    
    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        depth_data = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw.copy()).float()
        if not self.squeeze_channel:
            depth_raw_linear = depth_raw_linear.unsqueeze(0)  # [1, H, W]

        depth_data["depth_raw_linear"] = depth_raw_linear.clone()
        
        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled.copy()).float()
            if not self.squeeze_channel:
                depth_filled_linear = depth_filled_linear.unsqueeze(0)
            depth_data["depth_filled_linear"] = depth_filled_linear
        else:
            depth_data["depth_filled_linear"] = depth_raw_linear.clone()

        if self.kitti_bm_crop:
            depth_data["depth_raw_linear"] = self.kitti_benchmark_crop(depth_data["depth_raw_linear"])
        
        if self.train_crop is not None:
            depth_data = {
                k: self.kitti_train_crop(v) for k, v in depth_data.items()
            }
            # if depth_data["depth_filled_linear"].shape[1] == self.KB_CROP_HEIGHT and \
            #     depth_data["depth_filled_linear"].shape[2] == self.KB_CROP_WIDTH: # if filled_depth has been cropped
            #     depth_data["depth_raw_linear"] = self.kitti_benchmark_crop(depth_data["depth_raw_linear"])
            #     # print("OK.")
            # else:
            #     depth_data = {
            #         k: self.kitti_benchmark_crop(v) for k, v in depth_data.items()
            #     }
        return depth_data
    
    @staticmethod
    def kitti_benchmark_crop(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (torch.Tensor): Input image to be cropped.

        Returns:
            torch.Tensor: Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        else:
            raise NotImplementedError
        return out
    
    def kitti_train_crop(self, input_img):
        """
        Resize the input image so that the shorter side is 384, then randomly crop a 384x384 region.
        Args:
            input_img (torch.Tensor): Input image tensor (C x H x W or H x W).
        Returns:
            torch.Tensor: Cropped image tensor (C x 384 x 384 or 384 x 384).
        """
        TRAIN_CROP_SIZE = self.train_crop
        if input_img.ndim == 3:
            c, h, w = input_img.shape
        elif input_img.ndim == 2:
            h, w = input_img.shape
            c = None
        else:
            raise NotImplementedError("Input image must be 2D or 3D tensor.")
        
        if input_img.dtype not in (torch.float32, torch.float64):
            input_img = input_img.float()

        scale = TRAIN_CROP_SIZE / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        if c is not None:
            img_resized = F.interpolate(
                input_img.unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False).squeeze(0)
        else:
            img_resized = F.interpolate(
                input_img.unsqueeze(0).unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False).squeeze(0).squeeze(0)

        top = random.randint(0, new_h - TRAIN_CROP_SIZE)
        left = random.randint(0, new_w - TRAIN_CROP_SIZE)
        if c is not None:
            out = img_resized[:, top:top+TRAIN_CROP_SIZE, left:left+TRAIN_CROP_SIZE]
        else:
            out = img_resized[top:top+TRAIN_CROP_SIZE, left:left+TRAIN_CROP_SIZE]
        return out

    def _get_valid_mask(self, depth: torch.Tensor):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = super()._get_valid_mask(depth)  # [1, H, W]
        if self.valid_mask_crop is not None:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            gt_height, gt_width = eval_mask.shape

            if "garg" == self.valid_mask_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == self.valid_mask_crop:
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
        return valid_mask


class WeatherKITTIDepthMixedDataset(WeatherKITTIDepthDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mix_filenames = self.rgb_files + self.rain_files + self.raingan_files + \
                                self.fog1_files + self.fog2_files + self.snow_files + self.snowgan_files
        # self.rgb_files = self.rgb_files * (self.weather_opt.num_domains + 1)
        self.depth_files = self.depth_files * (self.weather_opt.num_domains + 1)
        self.aligned_depth_files = self.aligned_depth_files * (self.weather_opt.num_domains + 1)
        # print(len(self.mix_filenames), len(self.depth_files))

    def __len__(self):
        # mix_filenames
        return len(self.mix_filenames)
    
    def _get_data_path(self, index):
        rgb_rel_path = self.mix_filenames[index]
        # e.g., rgb/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            # e.g., depth/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
            depth_rel_path = self.depth_files[index]
            if self.has_filled_depth:          
                filled_rel_path = self.aligned_depth_files[index]  # e.g., 721.5377
        # print(rgb_rel_path, depth_rel_path, filled_rel_path)
        return rgb_rel_path, depth_rel_path, filled_rel_path


class WeatherKITTIDepthPairedDataset(WeatherKITTIDepthDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rgb_files = self.rgb_files * self.weather_opt.num_domains
        self.depth_files = self.depth_files * self.weather_opt.num_domains
        self.mix_filenames = self.rain_files + self.raingan_files + self.fog1_files \
                        + self.fog2_files + self.snow_files + self.snowgan_files
        print(len(self.mix_filenames), len(self.depth_files))
        
    def _get_data_path(self, index):
        rgb_rel_path = self.rgb_files[index]
        weather_rel_path = self.mix_filenames[index]
        # e.g., rgb/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            # e.g., depth/2011_10_03_drive_0034_sync/image_02/data/0000001499.png
            depth_rel_path = self.depth_files[index]
            if self.has_filled_depth:          
                filled_rel_path = self.filled[index]  # e.g., 721.5377
        return rgb_rel_path, weather_rel_path, depth_rel_path, filled_rel_path

    def _get_data_item(self, index):
        rasters = {}
        rgb_rel_path, weather_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)
        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        rasters.update(self._load_rgb_data(rgb_rel_path=weather_rel_path,
                                           dict_names=("weather_int", "weather_norm")))
        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, 
                filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]).clone()
            
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]).clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other