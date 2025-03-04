# ARM-HashNeRF-pytorch

This project implements Accelerated Ray Marching (ARM) in [HashNerf-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch), a pure PyTorch implementation of [Instant-NGP](https://github.com/NVlabs/instant-ngp). Instant-NGP drastically reduces (up to two orders of magnitude) the cost of training and evaluation of Neural Graphics Primitives that are parametrized by fully connected neural networks.

## ARM-HashNeRF vs Vanilla HashNeRF
Both the rendering time and quality of ARM-HashNeRF are compared against Vanilla HashNeRF. In all cases, **rendering time is reduced** while resulting in only a minimal decrease in redering quality (PSNR). For instance, in the 50K iterations comparison below, ARM-HashNeRF achieves 9.71% faster rendering compared to Vanilla HashNeRF, with only a 5.43% reduction in PSNR. Vanilla HashNeRF is on the left and ARM-HashNeRF on the right. All experiments were run using a single Tesla P100 GPU.


<!-- <video width="100%" controls>
  <source src="original-vs-art2_50K_default.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

https://raw.githubusercontent.com/jorgedanielrodrividal/ARM-HashNeRF-pytorch/main/original-vs-art2_50K_default.mp4




## Contents
- [Instructions](#instructions)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)


## Instructions

### 1. Download Dataset

The NeRF synthetic LEGO dataset is used in this project. Please download the preprocessed dataset from [here](https://drive.google.com/file/d/1spe2zFbqgz2Rt0fR1tJs5seJBxS0R-Sy/view) and place it in the `ARM-HashNeRF-pytorch/` directory.


### 2. Clone Repository

```
git clone git@github.com:jorgedanielrodrividal/ARM-HashNeRF-pytorch.git
```

### 3. Install custom vren library

```
pip install art/csrc/
```

### 4. Training

```
python run_arm_nerf.py --config configs/lego.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
```

---

## Citation

This project is mostly based on the amazing work of:

```
@misc{bhalgat2022hashnerfpytorch,
  title={HashNeRF-pytorch},
  author={Yash Bhalgat},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yashbhalgat/HashNeRF-pytorch/}},
  year={2022}
}
```

```
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

If you find this work useful, feel free to cite:

```
@misc{jorgedaniel2025armhashnerfpytorch,
  title={ARM-HashNeRF-pytorch},
  author={Jorge Daniel},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/jorgedanielrodrividal/ARM-HashNeRF-pytorch/}},
  year={2025}
}
```

---

## Acknowledgments
Big thanks to [Yash Bhalgat](https://github.com/yashbhalgat) for his enlightening HashNeRF project. Also thanks to the author of [ngp_pl](https://github.com/kwea123/ngp_pl), which served as a key inspiration for the ARM implementation.

