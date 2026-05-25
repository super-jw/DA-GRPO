# DA-GRPO: Data-Augmented GRPO for Robust GUI Agents

DA-GRPO is a reinforcement-learning framework that trains **vision-language GUI agents** (UI-TARS) to complete **long-horizon desktop tasks** while staying robust to **common environment corruptions**.

It combines three ingredients:

- **GRPO** (Group Relative Policy Optimization) end-to-end policy optimization, built on [EasyR1](https://github.com/hiyouga/EasyR1) / veRL.
- **Experience replay** for agentic rollouts, following [ARPO](https://github.com/dvlab-research/ARPO).
- **Data augmentation via environment corruptions** — during rollout, each parallel environment randomly applies a corruption (pop-ups, resolution change, visual marks, subtitles, multi-app clutter, …) drawn from [AgentHijack](https://github.com/tmlr-group/AgentHijack). Training under these perturbations makes the policy robust to the kinds of disruptions a real desktop throws at an agent.

---

## Overview

- **Distributed rollouts** across parallel [AgentHijack](https://github.com/tmlr-group/AgentHijack) / OSWorld Docker environments via Ray.
- **Multi-modal, long-horizon inputs**: histories of up to 15 screenshots + actions, processed end-to-end.
- **On-the-fly data augmentation**: corruptions are injected into the observation stream at `env.reset()` and after every step, so the agent learns to recover from degraded or adversarial UIs.

---

## Method: Data Augmentation via Environment Corruptions

The augmentation logic lives in `verl/trainer/noise.py` and `verl/trainer/perturb_utils.py` (adapted from AgentHijack), and is driven by `config/default.yaml`.

When each `EnvWorker` starts, it samples one corruption type for its environment (see `verl/trainer/gui_agent.py`). The active set used in training is:

| Corruption | Effect |
|------------|--------|
| `pop_ups` | Injects a fake pop-up / ad window over an empty region of the screen |
| `resolution` | Down-scales the observation resolution |
| `marks` | Scatters distractor marks (e.g. stars) across the screen |
| `subtitle` | Overlays subtitle-like text banners |
| `multi_apps` | Opens an extra unrelated application to clutter the desktop |

Additional corruptions (`accidental_touch`, `app_minimization`, `initialization_error`, `network_error`, `verification`, `wallpaper`) are implemented and configurable in `config/default.yaml`. Refer to the [AgentHijack paper](https://openreview.net/pdf?id=0H5Im3Xvuf) for the detailed parameterization.

---

## Installation

### 1. DA-GRPO

```bash
git clone https://github.com/super-jw/DA-GRPO.git
cd DA-GRPO

conda create -n da-grpo python=3.10
conda activate da-grpo

pip install -r requirements.txt
# optional, recommended for throughput:
# pip install flash_attn==2.7.4.post1
```

### 2. AgentHijack environment (`desktop_env`)

The training rollout imports `DesktopEnv` from AgentHijack's `desktop_env` package and reads tasks from its `evaluation_examples/`. Clone it **into the DA-GRPO root** and install it as an editable package:

```bash
git clone https://github.com/tmlr-group/AgentHijack.git
cd AgentHijack && pip install -e . && cd ..

# the Docker provider needs the docker SDK:
pip install docker
```

> AgentHijack is built on [OSWorld](https://github.com/xlang-ai/OSWorld). Follow OSWorld's Docker/VM setup guide to prepare the Docker image, Ubuntu VM data, and cache. We strongly recommend running a full AgentHijack evaluation **with Docker** once before training, to materialize the image / VM data / cache.

### 3. Link the environment cache

`gui_agent.py` expects the OSWorld cache at `cache_dirs/cache_0`:

```bash
mkdir -p cache_dirs
ln -s $(pwd)/AgentHijack/cache cache_dirs/cache_0
```

To run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## Training

### Configuration

- Base config: `examples/config.yaml`
- Per-run overrides + environment variables: `examples/agenthijack_da_grpo.sh`

Before launching, edit `examples/agenthijack_da_grpo.sh`:

- `MODEL_PATH` → path to `UI-TARS-1.5-7B` (or your own checkpoint to continue from).
- `WANDB_API_KEY` / `SWANLAB_API_KEY` → your own logging keys.
- `NUM_GPUS`, `NUM_ENVS`, `ROLLOUT_N` → match your hardware.
- `data.train_files` / `data.val_files` already point to
  `AgentHijack/evaluation_examples/test_success_uitars1.5_wo_impossible.json` (128 tasks).

### Start the Ray cluster

A Ray cluster must expose a custom `docker:<ip>` resource so env workers are pinned to nodes that can launch Docker environments.

```bash
RAY_PORT=2468
RAY_HEAD_IP=<YOUR_IP>
ray start --head --port=$RAY_PORT --resources='{"docker:'$RAY_HEAD_IP'": 128}'
```

(`start_ray.sh` is a convenience wrapper — set `RAY_HEAD_IP` inside it first.)

### Launch training

```bash
bash examples/agenthijack_da_grpo.sh
```

---

## Checkpoints

Training saves FSDP checkpoints under `trainer.save_checkpoint_path`. To export a HuggingFace model for serving / evaluation:

```bash
python scripts/model_merger.py --local_dir <path>/global_step_<N>/actor
```

---

## Evaluation

To benchmark a trained model on AgentHijack / OSWorld:

1. Serve the model with vLLM:

   ```bash
   # edit `model=` in start_server.sh to point at your merged checkpoint
   nohup bash start_server.sh &
   ```

2. Run AgentHijack's evaluation scripts (`run_multienv_uitars.py`, etc.) — see the
   [AgentHijack repository](https://github.com/tmlr-group/AgentHijack) for the full evaluation guide.

---

## Related Projects

- [AgentHijack](https://github.com/tmlr-group/AgentHijack) — Benchmark of computer-use-agent robustness to common environment corruptions; provides the `desktop_env` and corruption suite used here.
- [OSWorld](https://github.com/xlang-ai/OSWorld) — Realistic GUI environments for multimodal agents.
- [ARPO](https://github.com/dvlab-research/ARPO) — Agentic Replay Policy Optimization; source of the replay mechanism.
- [EasyR1](https://github.com/hiyouga/EasyR1) — Efficient, scalable multi-modality RL framework based on veRL.

---

## Citation

If you find this work useful, please cite AgentHijack:

```bibtex
@inproceedings{sun2026agenthijack,
  title     = {AgentHijack: Benchmarking Computer Use Agent Robustness to Common Environment Corruptions},
  author    = {Jingwei Sun and Jianing Zhu and Yuanyi Li and Tongliang Liu and Xia Hu and Bo Han},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=0H5Im3Xvuf}
}
```
