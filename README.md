# HyperGAN-CLIP: A Unified Framework for Domain Adaptation, Image Synthesis and Manipulation <br><sub>Official PyTorch Implementation of the SIGGRAPH Asia 2024 Paper</sub>
![Teaser image 1](srcs/teaser.png)
**HyperGAN-CLIP: A Unified Framework for Domain Adaptation, Image Synthesis and Manipulation**<br>
Abdul Basit Anees, Ahmet Canberk Baykal, Duygu Ceylan, Aykut Erdem, Erkut Erdem, Muhammed Burak Kızıl<br>
[![arXiv]()](https://arxiv.org/abs/xxxx.xxxxx)

*Generative Adversarial Networks (GANs), particularly StyleGAN and its variants, have demonstrated remarkable capabilities in generating highly realistic images. Despite their success, adapting these models to diverse tasks such as domain adaptation, reference-guided synthesis, and text-guided manipulation with limited training data remains challenging. Towards this end, in this study, we present a novel framework that significantly extends the capabilities of a pre-trained StyleGAN by integrating CLIP space via hyper-networks. This integration allows dynamic adaptation of StyleGAN to new domains defined by reference images or textual descriptions. Additionally, we introduce a CLIP-guided discriminator that enhances the alignment between generated images and target domains, ensuring superior image quality. Our approach demonstrates unprecedented flexibility, enabling text-guided image manipulation without the need for text-specific training data and facilitating seamless style transfer. Comprehensive qualitative and quantitative evaluations confirm the robustness and superior performance of our framework compared to existing methods.*

## Installation
### Clone the repository:
```shell
git clone https://github.com/basitanees/HyperGAN-CLIP.git
cd HyperGAN-CLIP
```

### Dependencies:
All python dependencies for defining the environment are provided in `./environment/environment.yml`.
```shell
conda env create -f environment/environment.yml
```

## Download pretrained models
Download files under `pretrained_models/`.
| Model | Description
| :--- | :----------
|[ffhq.pt](https://drive.google.com/uc?id=1XQabKtkpMltyZkFYidX4jd8Zrii5eTyI&export=download) | StyleGAN model pretrained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) with 1024x1024 output resolution.
|[ffhq_PCA.npz](https://drive.google.com/uc?id=13b81CBny0VgxWJWWEylNJkNbXuQ512ug&export=download) | PCA components of the pretrained StyleGAN(FFHQ) latent space.
|[ArcFace](https://drive.google.com/uc?id=1bwcB_AvbD0_qHGUoQCxzbp2wEurhjD4c&export=download) | Pretrained face recognition model to calculate identity loss.
|[afhqcat.pt](https://drive.google.com/uc?id=17K_U0IKaVKoQT4lJ6zf1h6ijfmrHSB7B&export=download) | StyleGAN model pretrained on [AFHQ_Cat](https://github.com/clovaai/stargan-v2) with 512x512 output resolution.
|[afhqcat_PCA.npz](https://drive.google.com/uc?id=1_JiWz-8eiki-LFFF0Aerf8GpM6mpjpYR&export=download) | PCA components of the pretrained StyleGAN(AFHQ_Cat) latent space.

## Training
Before training your model, put target images in the `target_data/{folder_name}/` directory and run
```shell
bash ./scripts/train_{mode}.sh
```
(Modify the script to include the corresponding paths)


## Acknowledgments
This code is borrowed from [DynaGAN](https://github.com/blueGorae/DynaGAN).


## Citation

```bibtex
@inproceedings{Kim2022DynaGAN,
    title     = {HyperGAN-CLIP: A Unified Framework for Domain Adaptation, Image Synthesis and Manipulation},
    author    = {Abdul Basit Anees and Ahmet Canberk Baykal and Duygu Ceylan and Aykut Erdem and Erkut Erdem and Muhammed Burak Kızıl},
    booktitle = {Proceedings of the ACM (SIGGRAPH Asia)},
    year      = {2024}
}
``` 