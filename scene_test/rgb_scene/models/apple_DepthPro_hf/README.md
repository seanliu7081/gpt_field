---
library_name: transformers
license: apple-amlr
tags:
- vision
- depth-estimation
pipeline_tag: depth-estimation
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

# DepthPro: Monocular Depth Estimation

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_pro_teaser.png)

This is the transformers version of DepthPro, a foundation model for zero-shot metric monocular depth estimation, designed to generate high-resolution depth maps with remarkable sharpness and fine-grained details. For the checkpoint compatible with the original codebase, please check [this repo](https://huggingface.co/apple/DepthPro).

## Table of Contents

- [DepthPro: Monocular Depth Estimation](#depthpro-monocular-depth-estimation)
  - [Table of Contents](#table-of-contents)
  - [Model Details](#model-details)
    - [Model Sources](#model-sources)
  - [How to Get Started with the Model](#how-to-get-started-with-the-model)
  - [Training Details](#training-details)
    - [Training Data](#training-data)
    - [Preprocessing](#preprocessing)
    - [Training Hyperparameters](#training-hyperparameters)
  - [Evaluation](#evaluation)
    - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Citation](#citation)
  - [Model Card Authors](#model-card-authors)

## Model Details

DepthPro is a foundation model for zero-shot metric monocular depth estimation, designed to generate high-resolution depth maps with remarkable sharpness and fine-grained details. It employs a multi-scale Vision Transformer (ViT)-based architecture, where images are downsampled, divided into patches, and processed using a shared Dinov2 encoder. The extracted patch-level features are merged, upsampled, and refined using a DPT-like fusion stage, enabling precise depth estimation.

The abstract from the paper is the following:

> We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions.

This is the model card of a 🤗 [transformers](https://huggingface.co/docs/transformers/index) model that has been pushed on the Hub.

- **Developed by:** Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun.
- **Model type:** [DepthPro](https://huggingface.co/docs/transformers/main/en/model_doc/depth_pro)
- **License:** Apple-ASCL

### Model Sources

<!-- Provide the basic links for the model. -->

- **HF Docs:** [DepthPro](https://huggingface.co/docs/transformers/main/en/model_doc/depth_pro)
- **Repository:** https://github.com/apple/ml-depth-pro
- **Paper:** https://arxiv.org/abs/2410.02073

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url = 'https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

inputs = image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)],
)

field_of_view = post_processed_output[0]["field_of_view"]
focal_length = post_processed_output[0]["focal_length"]
depth = post_processed_output[0]["predicted_depth"]
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = depth * 255.
depth = depth.detach().cpu().numpy()
depth = Image.fromarray(depth.astype("uint8"))
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The DepthPro model was trained on the following datasets:

![image/jpeg](assets/depth_pro_datasets.png)

### Preprocessing

Images go through the following preprocessing steps:
- rescaled by `1/225.`
- normalized with `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]`
- resized to `1536x1536` pixels

### Training Hyperparameters

![image/jpeg](assets/depth_pro_training_hyper_parameters.png)

## Evaluation

![image/png](assets/depth_pro_results.png)

### Model Architecture and Objective

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_pro_architecture.png)

The `DepthProForDepthEstimation` model uses a `DepthProEncoder`, for encoding the input image and a `FeatureFusionStage` for fusing the output features from encoder.

The `DepthProEncoder` further uses two encoders:
- `patch_encoder`
   - Input image is scaled with multiple ratios, as specified in the `scaled_images_ratios` configuration.
   - Each scaled image is split into smaller **patches** of size `patch_size` with overlapping areas determined by `scaled_images_overlap_ratios`.
   - These patches are processed by the **`patch_encoder`**
- `image_encoder`
   - Input image is also rescaled to `patch_size` and processed by the **`image_encoder`**

Both these encoders can be configured via `patch_model_config` and `image_model_config` respectively, both of which are separate `Dinov2Model` by default.

Outputs from both encoders (`last_hidden_state`) and selected intermediate states (`hidden_states`) from **`patch_encoder`** are fused by a `DPT`-based `FeatureFusionStage` for depth estimation.

The network is supplemented with a focal length estimation head. A small convolutional head ingests frozen features from the depth estimation network and task-specific features from a separate ViT image encoder to predict the horizontal angular field-of-view.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc{bochkovskii2024depthprosharpmonocular,
      title={Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
      author={Aleksei Bochkovskii and Amaël Delaunoy and Hugo Germain and Marcel Santos and Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
      year={2024},
      eprint={2410.02073},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.02073},
}
```

## Model Card Authors

[Armaghan Shakir](https://huggingface.co/geetu040)
