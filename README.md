# **DETR Object Detection**: End-to-End Object Detection with Transformers

---

Original Paper: [here](https://arxiv.org/pdf/2005.12872.pdf)    | Open source code : [here](https://github.com/facebookresearch/detr)  | Colab notebook [here](https://colab.research.google.com/github/ashishpatel26/Facebook-AI-DEtection-TRansformer-DETR-Object-Detection/blob/master/Facebook_AI_Detection_Transformer_detr.ipynb) 

---

PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer). We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

[![DETR](https://github.com/facebookresearch/detr/raw/master/.github/DETR.png)](https://github.com/facebookresearch/detr/blob/master/.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

# My Experiment on Model

![hotballon](D:\Ashish\github\Facebook-AI-DEtection-TRansformer-DETR-Object-Detection\images\hotballon.png)

| ![hotballon](D:\Ashish\github\Facebook-AI-DEtection-TRansformer-DETR-Object-Detection\images\hotballon.png) | ![streetlight](D:\Ashish\github\Facebook-AI-DEtection-TRansformer-DETR-Object-Detection\images\streetlight.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![carplane](D:\Ashish\github\Facebook-AI-DEtection-TRansformer-DETR-Object-Detection\images\carplane.png) | ![workdesk](D:\Ashish\github\Facebook-AI-DEtection-TRansformer-DETR-Object-Detection\images\workdesk.png) |



# Model Zoo

We provide baseline DETR and DETR-DC5 models, and plan to include more in future. AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images, with torchscript transformer.

|      | name     | backbone | schedule | inf_time | box AP | url                                                          | size  |
| ---- | -------- | -------- | -------- | -------- | ------ | ------------------------------------------------------------ | ----- |
| 0    | DETR     | R50      | 500      | 0.036    | 42.0   | [download](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) | 159Mb |
| 1    | DETR-DC5 | R50      | 500      | 0.083    | 43.3   | [download](https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth) | 159Mb |
| 2    | DETR     | R101     | 500      | 0.050    | 43.5   | [download](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth) | 232Mb |
| 3    | DETR-DC5 | R101     | 500      | 0.097    | 44.9   | [download](https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth) | 232Mb |

COCO val5k evaluation results can be found in this [gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).

COCO panoptic val5k models:

|      | name     | backbone | box AP | segm AP | PQ   | url                                                          | size  |
| ---- | -------- | -------- | ------ | ------- | ---- | ------------------------------------------------------------ | ----- |
| 0    | DETR     | R50      | 38.8   | 31.1    | 43.4 | [download](https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth) | 165Mb |
| 1    | DETR-DC5 | R50      | 40.2   | 31.9    | 44.6 | [download](https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth) | 165Mb |
| 2    | DETR     | R101     | 40.1   | 33      | 45.1 | [download](https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth) | 237Mb |

The models are also available via torch hub, to load DETR R50 with pretrained weights simply do:

```
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
```

# Usage

There are no extra compiled components in DETR and package dependencies are minimal, so the code is very simple to use. We provide instructions how to install dependencies via conda. First, clone the repository locally:

```
git clone https://github.com/facebookresearch/detr.git
```

Then, install PyTorch 1.5+ and torchvision 0.6+:

```
conda install -c pytorch pytorch torchvision
```

Install pycocotools (for evaluation on COCO) and scipy (for training):

```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:

```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from [http://cocodataset.org](http://cocodataset.org/#download). We expect the directory structure to be the following:

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training

To train baseline DETR on a single node with 8 gpus for 300 epochs run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```

A single epoch takes 28 minutes, so 300 epoch training takes around 6 days on a single machine with 8 V100 cards. To ease reproduction of our results we provide [results and training logs](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f) for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone. Horizontal flips, scales an crops are used for augmentation. Images are rescaled to have min size 800 and max size 1333. The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

## Evaluation

To evaluate DETR R50 on COCO val5k with a single GPU run:

```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```

We provide results for all DETR detection models in this [gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918). Note that numbers vary depending on batch size (number of images) per GPU. Non-DC5 models were trained with batch size 2, and DC5 with 1, so DC5 models show a significant drop in AP if evaluated with more than 1 image per GPU.

## Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

Train baseline DETR-6-6 model on 4 nodes for 300 epochs:

```python
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

# License

DETR is released under the Apache 2.0 license. Please see the [LICENSE](https://github.com/facebookresearch/detr/blob/master/LICENSE) file for more information**.**

**Note :  This Repository is for Practice purpose Only. All the Right is reserved by Original Author at Facebook AI.**

***Thanks For Reading***

---

