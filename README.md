# Vision Transformers for Dense Prediction
修改版仓库  
This repository contains code and models for our [paper](https://arxiv.org/abs/2103.13413):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


### Setup 

1) Download the model weights and place them in the `weights` folder:

Monodepth:
- [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)
- [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt), [Mirror](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view?usp=sharing)

  
2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5

### Usage 

1. Place one or more input images in the folder `input`.

2. Run a monocular depth estimation model:

    ```shell
    python run_monodepth.py
    ```
    跑训练和测试，使用`dpt_hybrid` 和 `dpt_large`  
    Use the flag `-t` to switch between different models. Possible options are `dpt_hybrid` (default) and `dpt_large`.  

3. The results are written to the folder `output_monodepth`.  

<!-- **Additional models:**

- Monodepth finetuned on KITTI: [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) [Mirror](https://drive.google.com/file/d/1-oJpORoJEdxj4LTV-Pc17iB-smp-khcX/view?usp=sharing)
- Monodepth finetuned on NYUv2: [dpt_hybrid_nyu-2ce69ec7.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) [Mirror](https\://drive.google.com/file/d/1NjiFw1Z9lUAfTPZu4uQ9gourVwvmd58O/view?usp=sharing) -->

Run with 

```shell
python run_monodepth.py -t [dpt_hybrid|dpt_large] 
```
-----------
### Evaluation for KITTI

* Remove images from `/input/` and `/output_monodepth/` folders
* Download `kitti_eval_dataset.zip` https://drive.google.com/file/d/1GbfMGuwg2VS06Vl75-_tB5FDj9EOrjl0/view?usp=sharing and unzip it in the `/input/` folder (or follow this repository https://github.com/cogaplex-bts/bts to get RGB and Depth images from list [eigen_test_files_with_gt.txt](https://github.com/cogaplex-bts/bts/blob/master/train_test_inputs/eigen_test_files_with_gt.txt) )
* Download [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) model and place it in the `/weights/` folder
* Download [eval_with_pngs.py](https://raw.githubusercontent.com/cogaplex-bts/bts/5a55542ebbe849eb85b5ce9592365225b93d8b28/utils/eval_with_pngs.py) in the root folder
--- 
#### Inference
* `python run_monodepth.py --model_type dpt_hybrid_kitti --kitti_crop --absolute_depth`
#### Evaluation
* `python ./eval_with_pngs.py --pred_path ./output_monodepth/ --gt_path ./input/gt/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --garg_crop --do_kb_crop`

Result:
```
Evaluating 697 files
GT files reading done
45 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.959,   0.995,   0.999,   0.062,   0.222,   2.575,   0.092,   8.282,   0.027
Done.
```

-----------
### Citation

Please cite our papers if you use this code or any of the models. 
```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
------------------------
### Acknowledgements

Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). We'd like to thank the authors for making these libraries available.
------------------------
### License 

MIT License 
