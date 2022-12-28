[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)

# Noise-aware Learning from Web-crawled Image-Text Data for Image Captioning
This is an official PyTorch Implementation of **Noise-aware Learning from Web-crawled Image-Text Data for Image Captioning** [[arxiv]](https://arxiv.org/abs/)
  - The method is a novel image captioning framework mainly for web-crawled image-text data.
  To be specific, the method addresses the noise issue in the web-crawled data by learning a quality controllable captioning model.
  The model is learned using alignment levels of the image-text pairs as an additional control signal during training.
  Such alignment-conditioned training allows the model to generate high-quality captions of well-aligned by simply setting the control signal to desired alignment level at inference time.

<p align="center"><img width="100%" src="./imgs/noc_framework.png"></p>

# Code Release
The code will be released soon.

# Citation
If you find this code helpful for your research, please cite our paper.
```
@article{kang2022noise,
  title={Noise-aware Learning from Web-crawled Image-Text Data for Image Captioning},
  author={Kang, Woo Young and Mun, Jonghwan and Lee, SungJun and Roh, Byungseok},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

# Contact for Issues
Woo Young Kang, [edwin.kang@kakaobrain.com](edwin.kang@kakaobrain.com)  
Jonghwan Mun, [jason.mun@kakaobrain.com](jason.mun@kakaobrain.com)  


# License
This project is licensed under the terms of the [Apache License 2.0](./LICENSE).
Copyright 2022 [Kakao Brain Corp](https://www.kakaobrain.com). All Rights Reserved.
