# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-Nav is a vision-language navigation (VLN) project that combines visual perception with language models for autonomous navigation in 3D environments. The system uses the Habitat simulator for embodied AI tasks and integrates large language models (LLMs) for instruction comprehension and decision-making.

## Commands

### Running the Main Navigation System
```bash
# Full evaluation (100 episodes)
./run.sh  # Edit script to set episodes_to_load=100, exp_name="eval"

# Debug mode (20 episodes)
./run.sh  # Default configuration

# Manual execution with custom parameters
python run.py \
  --exp_name <experiment_name> \
  --exp-config run_OpenNav.yaml \
  --llm <model_name> \
  --api_key <api_key> \
  --episodes_to_load <num_episodes>
```

### GPU Configuration
- Set `CUDA_VISIBLE_DEVICES` in run.sh to select GPU
- Configure `SIMULATOR_GPU_IDS`, `TORCH_GPU_ID`, and `TORCH_GPU_IDS` in the command flags

## Architecture

### Core Components

1. **Navigation System** (`vlnce_baselines/`)
   - `common/navigator/spatialNavigator.py`: Main navigation logic with Open_Nav class
   - `common/base_il_trainer_llm.py`: Base trainer for LLM-based navigation
   - `models/Policy_ViewSelection.py`: Policy for view selection
   - `ss_trainer.py`: Schedule sampler trainer

2. **Spatial Perception** (`SpatialBot3B/`)
   - Bunny model for spatial understanding and depth perception
   - Configuration in `config.json` and model weights in `.safetensors` files

3. **Visual Recognition** (`recognize_anything/`)
   - RAM (Recognize Anything Model) for object detection
   - Pretrained models in `pretrained/` directory

4. **Habitat Extensions** (`habitat_extensions/`)
   - Custom sensors, measures, and task configurations
   - Task config: `config/vlnce_task.yaml`

### Key Classes and Functions

- `Open_Nav` class: Orchestrates navigation with methods for:
  - `get_actions()`: Extract actions from instructions
  - `get_landmarks()`: Identify landmarks from actions
  - `observe_environment()`: Process visual observations
  - `move_to_next_vp()`: Decide next viewpoint
  - `estimate_completion()`: Check if goal is reached

### Data Flow

1. Instructions are parsed to extract actions and landmarks
2. Visual observations from multiple viewpoints are processed
3. Spatial reasoning model analyzes depth and spatial relationships
4. LLM makes navigation decisions based on history and observations
5. Actions are executed in the Habitat simulator

## Configuration

### Main Config Files
- `run_OpenNav.yaml`: Main experiment configuration
- `habitat_extensions/config/vlnce_task.yaml`: Task-specific settings
- `vlnce_baselines/config/default.py`: Default configurations

### Key Parameters
- `MAX_EPISODE_STEPS`: Maximum steps per episode (default: 20)
- `SUCCESS_DISTANCE`: Distance threshold for success (3.0 meters)
- `RGB_SENSOR`: 224x224 resolution, 90° HFOV
- `DEPTH_SENSOR`: 256x256 resolution for DDPPO ResNet

## Dataset Structure

- **Scenes**: MP3D dataset in `data/scene_datasets/mp3d/`
- **Episodes**: R2R-CE dataset in `data/datasets/R2R_VLNCE_v1-2_preprocessed/`
- **Pretrained Models**: `data/pretrained_models/`
- **Results**: `logs/eval_results/`
- **Checkpoints**: `logs/checkpoints/`

## LLM Integration

The system supports multiple LLMs through the API interface:
- GPT-4 models (e.g., gpt-4o-2024-08-06)
- Local models via transformers (e.g., Qwen/Qwen2.5-1.5B)

API calls are made through `vlnce_baselines/common/navigator/api.py`

## Important Notes

- The system uses multimodal inputs (RGB, depth, instructions)
- Navigation decisions combine visual perception with language understanding
- History tracking maintains context across navigation steps