# ThreeStepNav

Official implementation of **[Three-Step Nav: A Hierarchical Global-Local Planner for Zero-Shot Vision-and-Language Navigation](https://arxiv.org/abs/2604.26946)**.

Zero-shot **Vision-and-Language Navigation (VLN)** agents powered by multimodal large language models still tend to drift off course, halt prematurely, and achieve low overall success rates. Three-Step Nav is a hierarchical global-local planner that counteracts these failures with a three-view protocol:

1. **Look Forward** — extract global landmarks from the instruction and sketch a coarse plan.
2. **Look Now** — align the current visual observation with the next sub-goal for fine-grained guidance.
3. **Look Backward** — audit the trajectory so far to correct accumulated drift before stopping.

The planner requires no gradient updates or task-specific fine-tuning and drops into existing VLN pipelines with minimal overhead. This repo evaluates it in continuous environments (VLN-CE) on the [Habitat](https://aihabitat.org/) simulator, zero-shot on R2R-CE and RxR-CE, with standard VLN metrics.

## Supported LLMs

| Type | Models | Access |
|---|---|---|
| OpenAI | `gpt-4o-2024-08-06`, `gpt-5-2025-08-07` | `OPENAI_API_KEY` env var (or `--api_key`) |
| Local / open-source | `llama3.2-vision:90b`, `Qwen/Qwen2-72B`, `qwen3-vl:32b`, `qwen2.5vl:72b` | any OpenAI-compatible server (e.g. [Ollama](https://ollama.com/)), via `OLLAMA_BASE_URL` |

## Installation

1. Install `habitat-sim` and `habitat-lab` following the [VLN-CE setup guide](https://github.com/jacobkrantz/VLN-CE#setup) (this project uses the same simulator stack and task definitions).
2. Install the Python dependencies used by this repo:

```bash
pip install torch torchvision transformers openai tenacity langchain fastdtw
# For the visualization dashboard:
pip install streamlit plotly pandas matplotlib opencv-python Pillow
```

## Data preparation

| Asset | Location | Source |
|---|---|---|
| Matterport3D scenes | `data/scene_datasets/mp3d/` | Request access at [Matterport3D](https://niessner.github.io/Matterport/), then use `data/scene_datasets/download_mp.py` (note: this official script requires Python 2) |
| R2R-CE episodes (100-episode `val_unseen` subset) | `data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/` | **included in this repo** |
| DDPPO depth encoder `gibson-2plus-resnet50.pth` | `data/pretrained_models/ddppo-models/` | [DDPPO model zoo](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/rl/ddppo) (as used by VLN-CE) |
| Waypoint predictor `check_val_best_avg_wayscore` | `waypoint_prediction/checkpoints/` | [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN) (candidate waypoint predictor) |
| *(optional)* RAM checkpoint `ram_swin_large_14m.pth` | `recognize_anything/pretrained/` (or `RAM_CKPT_PATH`) | [Recognize Anything](https://github.com/xinyu1205/recognize-anything) |
| *(optional)* SpatialBot-3B weights | `SpatialBot3B/` (or `SPATIALBOT_PATH`) | [SpatialBot-3B](https://huggingface.co/RussRobin/SpatialBot-3B) |

The optional RAM / SpatialBot models power the spatial-perception descriptions in the **Look Now** step.

## Configuration

Environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required for GPT models) | — |
| `OLLAMA_BASE_URL` | OpenAI-compatible endpoint for local models | `http://localhost:11434/v1` |
| `RAM_CKPT_PATH` | RAM checkpoint path | `<repo>/recognize_anything/pretrained/ram_swin_large_14m.pth` |
| `SPATIALBOT_PATH` | SpatialBot-3B model directory | `<repo>/SpatialBot3B` |
| `OPENNAV_LOGS_DIR` | Logs directory read by the dashboard | `<repo>/logs` |

Main config files:

- `run_OpenNav.yaml` — experiment config (model, GPU ids, eval split, decision-agent meta-abilities)
- `habitat_extensions/config/vlnce_task.yaml` — task config (`MAX_EPISODE_STEPS: 8`, success distance 3.0 m, RGB 224×224 / depth 256×256 sensors)

## Inference

```bash
export OPENAI_API_KEY=<your-key>
./run.sh
```

`run.sh` wraps:

```bash
python run.py \
  --exp_name <experiment_name> \
  --exp-config run_OpenNav.yaml \
  --llm gpt-5-2025-08-07 \
  --episodes_to_load 100 \
  EVAL.SPLIT val_unseen
```

Results (per-episode metrics, aggregated stats, debug traces) are written to `logs/eval_results/<exp_name>/`.

## Visualization dashboard

Inspect navigation trajectories, per-step LLM interactions, and metrics in a Streamlit dashboard:

```bash
streamlit run view_navigation.py
# Point it at a non-default logs directory:
OPENNAV_LOGS_DIR=/path/to/logs streamlit run view_navigation.py
```

See `documentation/visualization_guide.md` for details.

## RxR-CE subset creation

To sample a 100-episode English subset from RxR and convert it to R2R format:

```bash
python data/datasets/create_rxr_dataset.py \
  --rxr-guide /path/to/val_unseen_guide.json \
  --rxr-gt /path/to/val_unseen_guide_gt.json
```

## Repository structure

```
run.py / run.sh                 # entry point
run_OpenNav.yaml                # main experiment config
vlnce_baselines/
  common/base_il_trainer_llm.py # evaluation loop
  common/navigator/             # LLM navigator: prompts, API clients, decision agent
  models/                       # view-selection policy + encoders
habitat_extensions/             # custom Habitat task, sensors, measures
waypoint_prediction/            # candidate waypoint predictor (TRM)
view_navigation.py              # Streamlit visualization dashboard
data/                           # scenes, episodes, pretrained models
```

## Acknowledgements

This project builds on [Open-Nav](https://arxiv.org/abs/2409.18794), [VLN-CE](https://github.com/jacobkrantz/VLN-CE), [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN), [Recognize Anything](https://github.com/xinyu1205/recognize-anything), and [SpatialBot](https://github.com/BAAI-DCAI/SpatialBot).

## License

[MIT](LICENSE)
