# FOR-age Tree Age Estimation

This repository contains the code for tree age estimation on the FOR-age dataset using a ForestFormer3D-based framework.

The repository is designed to keep the source code lightweight. Large point cloud files (`.laz`) and pretrained checkpoints (`.pth`) should be downloaded separately from Zenodo and placed under the expected folders described below.

---

## Repository Structure

```text
.
├── Dockerfile
├── configs
│   └── forestformer_age_pre.py
├── data
│   └── forage
│       ├── batch_load_forage_data.py
│       ├── load_forage_data.py
│       ├── meta_data
│       ├── test_data
│       └── train_val_data
├── oneformer3d
│   ├── forage_dataset.py
│   ├── spconv_unet.py
│   └── ...
├── replace_mmdetection_files
│   ├── base_model.py
│   ├── loops.py
│   └── transforms_3d.py
├── segmentator
├── tools
│   ├── create_data_forage.py
│   ├── test_forage_vote.py
│   └── ...
└── work_dirs
```

---

## Data and Checkpoints

The dataset and checkpoints are available at: [Zenodo link]

### Dataset

The `.laz` point cloud files should be downloaded from Zenodo and placed as follows:

```text
data/forage/train_val_data/*.laz
data/forage/test_data/*.laz
```

The expected data folders are:

```text
data/forage/meta_data
data/forage/train_val_data
data/forage/test_data
```

- `train_val_data`: training and validation `.laz` files.
- `test_data`: testing `.laz` files.
- `meta_data`: metadata files required by the FOR-age data loader.

### Checkpoints

The following pretrained or trained checkpoints should also be downloaded separately from Zenodo and placed under `work_dirs`:

```text
work_dirs/load_ff3d_forage/best_RMSE_epoch_1050_fixed.pth
work_dirs/scratch_forage/best_RMSE_epoch_1050_fix.pth
work_dirs/clean_forestformer/epoch_3000_fix.pth
```

Meanings of these checkpoints:

- `work_dirs/clean_forestformer/epoch_3000_fix.pth`: ForestFormer3D pretrained checkpoint used for initialization.
- `work_dirs/load_ff3d_forage/best_RMSE_epoch_1050_fixed.pth`: age estimation checkpoint trained/fine-tuned with ForestFormer3D initialization.
- `work_dirs/scratch_forage/best_RMSE_epoch_1050_fix.pth`: age estimation checkpoint trained from scratch.

---

## Environment Setup

### 1. Build Docker Image

From the project root:

```bash
cd /path/to/for_age_code
sudo docker build -t forage-image .
```

### 2. Run Docker Container

Example:

```bash
sudo docker run --gpus all \
  --shm-size=128g \
  -d \
  -p 127.0.0.1:49288:22 \
  -v /path/to/for_age_code:/workspace \
  --name forage-container \
  forage-image
```

Enter the running container:

```bash
sudo docker exec -it forage-container /bin/bash
```

Inside the container, the project should be available at:

```text
/workspace
```

---

## Dependency Fixes

Depending on the environment, some packages may need to be reinstalled manually.

### Torch Points Kernels

Test whether `torch_points_kernels` is correctly installed:

```bash
python -c "from torch_points_kernels import instance_iou; print('torch-points-kernels loaded successfully')"
```

If you see:

```text
ModuleNotFoundError: No module named 'torch_points_kernels.points_cuda'
```

reinstall it with:

```bash
pip uninstall torch-points-kernels -y
pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0
```

### Torch Cluster

```bash
pip uninstall torch-cluster -y
pip install torch-cluster --no-cache-dir --no-deps
```

### Replace Required MMEngine / MMDetection3D Files

The repository provides several replacement files under `replace_mmdetection_files`.

```bash
cp /workspace/replace_mmdetection_files/loops.py \
  /opt/conda/lib/python3.10/site-packages/mmengine/runner/

cp /workspace/replace_mmdetection_files/base_model.py \
  /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/

cp /workspace/replace_mmdetection_files/transforms_3d.py \
  /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/
```

If your Python environment is different, first check the installation path with:

```bash
pip show mmengine
pip show mmdet3d
```

---

## Data Preparation

### 1. Place Raw Data

After downloading the data from Zenodo, the folder should look like:

```text
/workspace/data/forage
├── meta_data
├── train_val_data
│   ├── xxx.laz
│   └── ...
└── test_data
    ├── xxx.laz
    └── ...
```

### 2. Install LAZ Dependencies

```bash
pip install laspy
pip install "laspy[lazrs]"
```

### 3. Load FOR-age Data

```bash
cd /workspace/data/forage
python batch_load_forage_data.py
```

This step converts the raw `.laz` files into the intermediate data format required by the training pipeline.

### 4. Create Training Info Files

Return to the project root:

```bash
cd /workspace
python tools/create_data_forage.py forage
```

---

## Training

Set the Python path:

```bash
export PYTHONPATH=/workspace
```

### Train from Scratch

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  /workspace/configs/forestformer_age_pre.py \
  --work-dir /workspace/work_dirs/scratch_forage
```

### Train with ForestFormer3D Pretrained Checkpoint

First, make sure the pretrained checkpoint exists:

```text
/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
```

Then set `load_from` in:

```text
/workspace/configs/forestformer_age_pre.py
```

For example:

```python
load_from = '/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth'
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  /workspace/configs/forestformer_age_pre.py \
  --work-dir /workspace/work_dirs/load_ff3d_forage
```

---

## Testing / Inference

### 1. Optional: Fix the Checkpoint for SpConv Compatibility


Before testing, convert the checkpoint if needed (If you use the provided `*_fix.pth` or `*_fixed.pth` checkpoints, this step can be skipped):

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/<output_folder_name>/best_RMSE_epoch_xxx.pth \
  --out-path /workspace/work_dirs/<output_folder_name>/best_RMSE_epoch_xxx_fix.pth
```

Example:

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/scratch_forage/best_RMSE_epoch_1050.pth \
  --out-path /workspace/work_dirs/scratch_forage/best_RMSE_epoch_1050_fix.pth
```

### 2. Configure Test Script

Open:

```text
/workspace/tools/test_forage_vote.py
```

Modify the following variables according to your experiment:

```python
CONFIG_FILE_PATH = '/workspace/configs/forestformer_age_pre.py'
BASE_OUTPUT_DIR = '/workspace/work_dirs/<output_folder_name>'
MODEL_PATH = '/workspace/work_dirs/<output_folder_name>/<checkpoint_name>.pth'
```

In `oneformer3d/oneformer3d.py`, in the `predict` function of `ForestFormerDownstream_Age`, set:

```python
is_test = True
```

### 3. Run Testing

```bash
python tools/test_forage_vote.py
```

---

## Plot Prediction Results

The script `tools/plot_age_dots.py` plots predicted age against ground-truth age and reports common regression metrics.

Usage:

```bash
python /workspace/tools/plot_age_dots.py <input_csv> <output_png>
```

Example:

```bash
python /workspace/tools/plot_age_dots.py \
  /workspace/work_dirs/scratch_forage/predictions.csv \
  /workspace/work_dirs/scratch_forage/age_dots.png
```

---

## TensorBoard Visualization

```bash
tensorboard \
  --logdir=/workspace/work_dirs/<output_folder_name>/vis_data/ \
  --host=0.0.0.0 \
  --port=6006
```


