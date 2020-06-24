# Seq-NMS for Video Object Detection
This project provides a python implementation of [Seq-NMS] (https://arxiv.org/abs/1602.08465) post-processing algorithm for video object detection.

## Setup Details 
This repository is developed and tested with python 3.7.3. Setup requires and is tested with cython (0.29.20) and numpy (1.18.5)

Run the following command to build and compile the cython file compute_overlap.pyx: 
```bash
python setup.py build_ext --inplace
```


