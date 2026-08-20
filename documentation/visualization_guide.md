# Open-Nav Visualization Dashboard Guide

## Overview
The Open-Nav Visualization Dashboard is a Streamlit-based web application for analyzing Vision-Language Navigation experiment results. It provides interactive visualizations and metrics for evaluating navigation performance.

## Data Sources

### 1. Debug Information File (`debug.json`)
Location: `/logs/eval_results/{experiment_name}/debug.json`

**Data Structure:**
```json
{
    "experiment_name": "val_unseen",
    "episodes": [
        {
            "episode_id": 7,
            "scene_id": "data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
            "instruction": "Go straight past the pool. Walk between the bar and chairs...",
            "steps": [
                {
                    "step_index": 1,
                    "current_action": "Go straight past the pool",
                    "viewpoints": {
                        "0": {
                            "rgb_path": "step_1_vp_0_rgb.jpg",
                            "depth_path": "step_1_vp_0_depth.jpg",
                            "is_chosen": false
                        },
                        "1": {
                            "rgb_path": "step_1_vp_1_rgb.jpg",
                            "depth_path": "step_1_vp_1_depth.jpg",
                            "is_chosen": true
                        }
                        // ... more viewpoints
                    },
                }
                // ... more steps
            ]
        }
        // ... more episodes
    ]
}
```

**Fields:**
- `experiment_name`: Name of the evaluation split (e.g., "val_unseen", "val_seen")
- `episodes`: Array of episode information
  - `episode_id`: Unique identifier for the episode (integer)
  - `scene_id`: Path to the 3D scene file in MP3D dataset
  - `instruction`: Natural language navigation instruction
  - `steps`: Array of navigation steps
    - `step_index`: Step number (1-based indexing)
    - `current_action`: Current sub-instruction being executed
    - `viewpoints`: Dictionary of all viewpoints with their images
      - `rgb_path`: Path to RGB image relative to episode_images folder
      - `depth_path`: Path to depth image relative to episode_images folder
      - `is_candidate`: Boolean indicating if viewpoint was a candidate
      - `is_chosen`: Boolean indicating if viewpoint was chosen
    - `candidate_viewpoints`: List of viewpoint IDs that were candidates
    - `chosen_viewpoint`: ID of the viewpoint that was chosen

### 2. Evaluation Results File (`stats_ep_ckpt_*.json`)
Location: `/logs/eval_results/{experiment_name}/stats_ep_ckpt_{split}_r{rank}_w{world_size}.json`

**Data Structure:**
```json
{
    "7": {
        "steps_taken": 6.0,
        "distance_to_goal": 3.8004887104034424,
        "success": 0.0,
        "oracle_success": 0.0,
        "path_length": 10.709141247951363,
        "collisions": 0.09302325581395349,
        "spl": 0.0,
        "ndtw": 0.6775294205782666
    },
    // ... more episodes
}
```

**Metrics:**
- `steps_taken`: Number of navigation steps executed
- `distance_to_goal`: Final distance to target location (meters)
- `success`: Binary success indicator (1.0 if goal reached within 3m)
- `oracle_success`: Success if agent ever came within 3m of goal
- `path_length`: Total distance traveled (meters)
- `collisions`: Collision rate during navigation
- `spl`: Success weighted by Path Length (efficiency metric)
- `ndtw`: Normalized Dynamic Time Warping (trajectory similarity metric)

## User Interface Options

### Sidebar Controls

1. **Experiment Selection**
   - Dropdown menu listing all available experiments
   - Experiments are folder names in `/logs/eval_results/`
   - Example: "first", "test_run", "full_eval"

2. **Episode Selection**
   - Dropdown menu showing all episode IDs from selected experiment
   - Episodes loaded from `debug.json`
   - Displays episode count, success rate, and oracle success rate

### Main View Tabs

The interface provides two main tabs:

1. **Episode View** - Individual episode analysis
2. **Experiment Summary** - Aggregate statistics

## Visualizations and Information

### Episode View Tab

**Episode Information Panel:**
- Episode ID (integer identifier)
- Scene ID (MP3D scene file path)
- Full instruction text

**Episode Metrics Display:**
- Success indicator (✅/❌)
- Oracle Success indicator (✅/❌)
- SPL (Success weighted by Path Length)
- NDTW (Normalized Dynamic Time Warping)
- Steps taken
- Distance to goal (meters)
- Path length (meters)
- Expandable detailed metrics table

**Step-by-Step Navigation:**
- Step selector slider (for multiple steps) or single step indicator
- Current action display for each step
- Viewpoint grid (4 columns) showing:
  - All viewpoints captured at each step
  - ✅ **Checkmark**: Chosen viewpoint selected by the agent
- RGB images
- Expandable depth images for each viewpoint
- Candidate viewpoints list
- Chosen viewpoint identifier

### Experiment Summary Tab

**Summary Statistics:**
- Total episodes evaluated
- Overall success rate (percentage)
- Oracle success rate (percentage)
- Average SPL across all episodes
- Average NDTW score
- Average steps taken
- Average distance to goal

**Visualizations:**

1. **Success Distribution Pie Chart**
   - Visual breakdown of successful vs failed episodes
   - Color-coded (green for success, red for failure)

2. **SPL Distribution Histogram**
   - Shows distribution of SPL scores across episodes
   - 20 bins for granular analysis
   - Helps identify performance patterns

3. **Distance to Goal Distribution Histogram**
   - Visualizes final distances to target
   - Useful for understanding near-misses
   - 20 bins for detailed distribution

4. **Full Episodes Table**
   - Sortable table with all episodes
   - Complete metrics for each episode
   - Success indicators with emoji formatting
   - Formatted numerical values for readability

## Usage Instructions

1. **Launch the Dashboard:**
   ```bash
   streamlit run view_navigation.py
   ```

2. **Select Experiment:**
   - Choose experiment from sidebar dropdown
   - Dashboard automatically loads `debug.json` and stats files

3. **Analyze Episodes:**
   - Select individual episodes for detailed view
   - Review instruction and performance metrics
   - Compare against experiment averages

4. **Review Summary:**
   - Switch to Experiment Summary tab
   - Analyze overall performance trends
   - Export data tables as needed

## Key Insights Available

- **Success Patterns**: Identify which types of instructions lead to success
- **Oracle vs Final Success**: Compare cases where agent reached goal but didn't stop
- **Efficiency Analysis**: SPL scores reveal navigation efficiency
- **Error Analysis**: Distance to goal shows how close failures were
- **Trajectory Quality**: NDTW indicates path following accuracy
- **Scene Complexity**: Compare performance across different scenes
- **Instruction Length**: Correlate instruction complexity with success
- **Decision Making**: Analyze viewpoint selection patterns at each step
- **Candidate Analysis**: Understand which viewpoints were considered vs chosen
- **Step-by-Step Progress**: Track navigation decisions throughout the episode

## Requirements

- Completed experiment with generated `debug.json`
- Stats files from evaluation (`stats_ep_ckpt_*.json`)
- Streamlit installation (`pip install streamlit`)
- Required Python packages: pandas, numpy, plotly, matplotlib