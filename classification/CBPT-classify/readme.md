# Installation

```
pytorch==1.7.1
torchvision==0.8.2
timm==0.3.2
apex==0.1
opencv-python==4.4.0.46
termcolor==1.1.0
yacs==0.1.8
```



# Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```



# Training from scratch

To train a  CBPT on ImageNet from scratch, run:

```
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]

```



# Evaluation

To evaluate a pre-trained CBPT  on ImageNet val, run:

```
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```



# Throughput

To measure the throughput, run:

```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --amp-opt-level O0
```

