
# A Minimal Implementation of Music tagging

0. Install dependencies

```bash
git clone https://github.com/qiuqiangkong/mini_music_tagging

# Install Python environment.
conda create --name music_tagging python=3.8

# Activate environment.
conda activate music_tagging

# Install Python packages dependencies.
sh env.sh

```

# Single GPU training.

Here is an example of Pytorch for single GPU training.

```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

# Multiple GPUs training.

For multiple GPUs training, Lightning Fabric is recommended due to its easy to use. Users just need to add less than 10 lines to the Pytorch code to train on multiple GPUs. Example:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_fabric.py
```

# Inference
```python
CUDA_VISIBLE_DEVICES=0 python inference.py
```