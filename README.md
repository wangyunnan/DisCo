### <div align="center"> Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation<div> 
<div align="center">
<div style="text-align: center;">
  <a href="https://arxiv.org/abs/2410.00447"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:DisCo&color=red&logo=arxiv"></a> &ensp;
  <a href="https://neurips.cc/virtual/2024/poster/92965"><img src="https://img.shields.io/badge/Project-Website-blue"></a> &ensp;
</div>
</div> 
</div>


## 📣 Update Log
- [2024.12.24] 🎉 Here comes DisCo, we release the training and evaluation code of DisCo. 


## 🪄✨ Abstract
<b>TL; DR: <font color="red">DisCo</font> leverage the scene graph, a powerful structured representation, for complex image generation.</b>
<details><summary>CLICK for the full abstract</summary>
There has been exciting progress in generating images from natural language or layout conditions. However, these methods struggle to faithfully reproduce complex scenes due to the insufficient modeling of multiple objects and their relationships. To address this issue, we leverage the scene graph, a powerful structured representation, for complex image generation. Different from the previous works that directly use scene graphs for generation, we employ the generative capabilities of variational autoencoders and diffusion models in a generalizable manner, compositing diverse disentangled visual clues from scene graphs. Specifically, we first propose a Semantics-Layout Variational AutoEncoder (SL-VAE) to jointly derive (layouts, semantics) from the input scene graph, which allows a more diverse and reasonable generation in a one-to-many mapping. We then develop a Compositional Masked Attention (CMA) integrated with a diffusion model, incorporating (layouts, semantics) with fine-grained attributes as generation guidance. To further achieve graph manipulation while keeping the visual content consistent, we introduce a Multi-Layered Sampler (MLS) for an "isolated" image editing effect. Extensive experiments demonstrate that our method outperforms recent competitors based on text, layout, or scene graph, in terms of generation rationality and controllability.
</details>


## ⚙️ Setup
Follow the following guide to set up the environment.
- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.2.1+cu11.8](https://pytorch.org/)
- Better create a virtual environment

Install the required dependencies by following the command.

1. git clone repo.
    ```
    git clone https://github.com/wangyunnan/DisCo.git
    cd DisCo
    ```

## 📖 Citation
Don't forget to cite this source if it proves useful in your research!
```bibtex
@article{wang2024scene,
    title={Scene graph disentanglement and composition for generalizable complex image generation},
    author={Wang, Yunnan and Li, Ziqiang and Zhang, Wenyao and Zhang, Zequn and Xie, Baao and Liu, Xihui and Zeng, Wenjun and Jin, Xin},
    journal={Advances in Neural Information Processing Systems (NeurlPS)},
    volume={37},
    pages={98478--98504},
    year={2024}
}
```

## License
This repository is released under the MiT license as found in the [LICENSE](LICENSE) file.