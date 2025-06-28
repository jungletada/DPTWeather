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

#### Inference

	python infer.py

#### Evaluation
	python eval.py


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
