# RD-VLA: Recurrent-Depth Vision-Language-Action Model

Implicit test-time compute scaling of VLA models via latent iterative reasoning.

RD-VLA introduces a weight-tied recurrent transformer core that performs iterative refinement in latent space, enabling adaptive test-time compute with constant memory footprint. The architecture decomposes the action head into three stages: **Prelude** (grounding), **Recurrent Core** (iterative refinement), and **Coda** (action projection).

**Paper:** [arXiv:2602.07845](https://arxiv.org/abs/2602.07845)

## Results

### LIBERO Benchmark

| Method | Params | Spatial | Object | Goal | Long | Avg. |
|--------|--------|---------|--------|------|------|------|
| RD-VLA (Fixed, Rec=12) | 0.5B | 92.0 | 99.0 | 96.0 | 84.8 | **93.0** |
| RD-VLA (Adaptive) | 0.5B | 88.6 | 98.8 | 96.8 | 85.8 | **92.5** |

### CALVIN ABC->D

| Method | 1 | 2 | 3 | 4 | 5 | Avg. len |
|--------|---|---|---|---|---|----------|
| RD-VLA | 91.4 | 79.5 | 67.9 | 54.9 | 45.3 | **3.39** |

## Setup

```bash
conda create -n rdvla python=3.10.16 -y
conda activate rdvla

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### VLM Backbone

```bash
cd pretrained_models
git lfs install
git clone https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b
cd ..
```

### LIBERO Data (for training)

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds libero
cd ..
```

The dataset directories are already named correctly (`libero_spatial_no_noops`, etc.).

### LIBERO Simulator (for evaluation)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Checkpoints

| Suite | Steps | Link |
|-------|-------|------|
| LIBERO-Spatial | 40k | [hqfang/12_24-24_24_Spatial_40k](https://huggingface.co/hqfang/12_24-24_24_Spatial_40k) |
| LIBERO-Object | 40k | [hqfang/12_24-24_24_Object_40k](https://huggingface.co/hqfang/12_24-24_24_Object_40k) |
| LIBERO-Goal | 60k | [hqfang/12_24-24_24_Goal_60k](https://huggingface.co/hqfang/12_24-24_24_Goal_60k) |
| LIBERO-Long | 75k | [hqfang/12_24-24_24_Long_75k](https://huggingface.co/hqfang/12_24-24_24_Long_75k) |

Download into `outputs/`:
```bash
huggingface-cli download hqfang/12_24-24_24_Spatial_40k --local-dir outputs/12_24-24_24_Spatial_40k
```

## Training

```bash
# Single GPU (24GB+)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  run.py --config configs/train/rdvla_spatial.yaml --mode train

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  run.py --config configs/train/rdvla_spatial.yaml --mode train
```

Training configs: `configs/train/rdvla_{spatial,object,goal,long}.yaml`

Override any config value via CLI:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  run.py --config configs/train/rdvla_spatial.yaml --mode train \
  --batch_size=4 --optimizer.learning_rate=1e-4
```

## Evaluation

### LIBERO

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/12_24-24_24_Spatial_40k \
  --task_suite_name libero_spatial \
  --use_recurrent True
```

### Adaptive Computation

RD-VLA supports adaptive stopping based on action convergence:

```bash
# Fixed iterations (default: 12)
--recurrence_strategy fixed --recurrent_num_iter 12

# KL divergence stopping (default threshold: 0.001, max 32 iters)
--recurrence_strategy kl_divergence --recurrence_kl_thresh 0.001 --recurrence_max_iter 32

# Cosine similarity stopping (default threshold: 0.999, max 32 iters)
--recurrence_strategy cosine_similarity --recurrence_cos_thresh 0.999 --recurrence_max_iter 32
```

### Adaptive Execution

Couple recurrence depth with action horizon:

```bash
# Binary threshold: fewer actions when convergence is fast (uncertain), more when slow (confident)
--adaptive_exec True --adaptive_exec_threshold 4 --adaptive_exec_low 4 --adaptive_exec_high 8

# Dynamic 4-bucket system (2/4/6/8 actions) based on running mean/std of iterations
--dynamic_exec True --dynamic_exec_warmup 5

# Linear decay: maps iteration count to action horizon (few iters → 8, many iters → down to 2)
--use_linear_decay_horizon True
```

### Logging & Visualization

```bash
# Save per-episode stats (iters, success) to JSON
--json_log_file results.json

# Videos automatically include thinking steps overlay when using recurrence
```

### CALVIN ABC->D

```bash
python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint outputs/CALVIN-ABC
```

## Architecture

```
VLM (Qwen2.5-0.5B + LoRA) -> [h_vis, h_lat, p]
                                    |
                              Prelude (P_phi)
                              Cross-attend to h^(12)
                              -> S_pre
                                    |
                              Scratchpad Init
                              S_0 ~ TruncNormal(0, gamma * sigma)
                                    |
                              Recurrent Core (R_theta)  [weight-tied, N iterations]
                              x_k = RMSNorm(W_adapt[S_{k-1}; S_pre])
                              S_k = R_theta(x_k, [h^(24); p])
                                    |
                              Coda (C_psi)
                              a = W_out * RMSNorm(C_psi(S_r, [h^(24); p]))
```

Training uses log-normal Poisson sampling of iterations (mu=32) with TBPTT (d=8 gradient steps).

## Project Structure

```
prismatic/models/action_heads.py            # Core RD-VLA action head (RecurrentLayer, VLARecurrent, ActionHeadRecurrent)
prismatic/extern/hf/modeling_prismatic.py   # VLM backbone integration
vla-scripts/finetune.py                     # Training script
experiments/robot/libero/run_libero_eval.py # LIBERO evaluation
experiments/robot/openvla_utils.py          # Model loading and inference
configs/                                    # YAML training/eval configs
```

## Citation

```bibtex
@article{tur2025rdvla,
  title={Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning},
  author={Tur, Yalcin and Naghiyev, Jalal and Fang, Haoquan and Tsai, Wei-Chuan and Duan, Jiafei and Fox, Dieter and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2602.07845},
  year={2025}
}
```

## Acknowledgments

Built upon [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), and [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter).
