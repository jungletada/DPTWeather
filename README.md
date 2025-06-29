# Vision Transformers for Dense Prediction
修改版仓库  
This repository contains code and models for our [paper](https://arxiv.org/abs/2103.13413):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


### Setup 

1) Download the model weights and place them in the `weights` folder:

Monodepth:
* Download [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) model and place it in the `/weights/` folder
* Download [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt) model and place it in the `/weights/` folder


  
2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

-----------
### Evaluation for KITTI (Fully supervised)

#### Inference 生成推理的深度图（Inverse），保存为npy文件

	python infer.py

`--output_path`: 指定输出`.npy`深度图的路径，默认为`output/prediction`

`--model_weights`: 指定模型权重的文件夹，默认采用`weights/dpt_hybrid_kitti-cb926ef4.pt`

`--model_type`: 模型的类型，默认采用在kitti上微调的`dpt_hybrid_kitti`；  其他可选模型：`dpt_hybrid`, `dpt_large` （需要下载相应的权重）

#### Evaluation 根据上面生成的深度图，进行评价
	python eval.py
`--prediction_dir`: 指定之前已经生成的`.npy`深度图的路径，默认为`output/prediction`

`--dataset_config`: 数据集的配置文件，默认采用测试集`config/dataset_depth/data_kitti_eigen_test.yaml`

`--base_data_dir`: 数据集的根目录，默认`data/kitti`

`--output_dir`: 评价指标的输出目录，默认`output/eval_metric`

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
