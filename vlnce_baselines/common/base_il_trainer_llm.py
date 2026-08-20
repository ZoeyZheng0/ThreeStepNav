import json
import sys
import jsonlines
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI
import cv2
import base64
import io

# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import *
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

# Repository root, used to anchor checkpoint/cache/log paths independently of the CWD
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    if isinstance(image_array, np.ndarray):
        # Convert to PIL Image
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB image
            pil_image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
        elif len(image_array.shape) == 2:
            # Grayscale image
            pil_image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        else:
            # Other formats, convert to RGB
            pil_image = Image.fromarray(image_array.astype(np.uint8))
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    return None

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                os.path.join(REPO_ROOT, 'waypoint_prediction/checkpoints/check_val_best_avg_wayscore'),
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {}
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            if 'rgb' in key:
                # Convert numpy array to PIL Image for rgb images
                rgb_image_dict[str(rgb_index)] = Image.fromarray(observations[key], mode="RGB")
                rgb_index += 1
            if 'depth' in key:
                # Process depth images
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                else:
                    depth_map = observations[key]
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                depth_image_dict[str(depth_index)] = Image.fromarray(depth_img)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index],
                'base64': image_to_base64(np.array(rgb_image_dict[index]))
            }

        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        angles = batch_angles[-1]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            if 0 < angle_deg <= 30:
                waypoint_images['1'] = image_dict['1']
                waypoint_distances['1'] = batch_distance[angle_idx]
                waypoint_radius['1'] = angles[angle_idx]
            elif 30 < angle_deg <= 60:
                waypoint_images['2'] = image_dict['2']
                waypoint_distances['2'] = batch_distance[angle_idx]
                waypoint_radius['2'] = angles[angle_idx]
            elif 60 < angle_deg <= 90:
                waypoint_images['3'] = image_dict['3']
                waypoint_distances['3'] = batch_distance[angle_idx]
                waypoint_radius['3'] = angles[angle_idx]
            elif 90 < angle_deg <= 120:
                waypoint_images['4'] = image_dict['4']
                waypoint_distances['4'] = batch_distance[angle_idx]
                waypoint_radius['4'] = angles[angle_idx]
            elif 120 < angle_deg <= 150:
                waypoint_images['5'] = image_dict['5']
                waypoint_distances['5'] = batch_distance[angle_idx]
                waypoint_radius['5'] = angles[angle_idx]
            elif 150 < angle_deg <= 180:
                waypoint_images['6'] = image_dict['6']
                waypoint_distances['6'] = batch_distance[angle_idx]
                waypoint_radius['6'] = angles[angle_idx]
            elif 180 < angle_deg <= 210:
                waypoint_images['7'] = image_dict['7']
                waypoint_distances['7'] = batch_distance[angle_idx]
                waypoint_radius['7'] = angles[angle_idx]
            elif 210 < angle_deg <= 240:
                waypoint_images['8'] = image_dict['8']
                waypoint_distances['8'] = batch_distance[angle_idx]
                waypoint_radius['8'] = angles[angle_idx]
            elif 240 < angle_deg <= 270:
                waypoint_images['9'] = image_dict['9']
                waypoint_distances['9'] = batch_distance[angle_idx]
                waypoint_radius['9'] = angles[angle_idx]
            elif 270 < angle_deg <= 300:
                waypoint_images['10'] = image_dict['10']
                waypoint_distances['10'] = batch_distance[angle_idx]
                waypoint_radius['10'] = angles[angle_idx]
            elif 300 < angle_deg <= 330:
                waypoint_images['11'] = image_dict['11']
                waypoint_distances['11'] = batch_distance[angle_idx]
                waypoint_radius['11'] = angles[angle_idx]
            else:
                waypoint_images['0'] = image_dict['0']  
                waypoint_distances['0'] = batch_distance[angle_idx]
                waypoint_radius['0'] = angles[angle_idx]
                
        return waypoint_images, waypoint_radius, waypoint_distances
    
    def _create_reverse_action(self, action):
        """
        Create a reverse action to undo the given action.
        Args:
            action: The original action to reverse
        Returns:
            The reverse action
        """
        if action['action']['action'] == 4:  # Move action
            # For move actions, reverse the angle by adding 180 degrees (π radians)
            original_angle = action['action']['action_args']['angle']
            original_distance = action['action']['action_args']['distance']
            
            # Reverse the angle (add π radians)
            reverse_angle = original_angle + math.pi
            # Normalize to [-π, π]
            reverse_angle = (reverse_angle + math.pi) % (2 * math.pi) - math.pi
            
            return {
                'action': {
                    'action': 4,
                    'action_args': {
                        'angle': reverse_angle,
                        'distance': original_distance,
                    }
                }
            }
        else:
            # For other actions, just return the same action (no reverse needed)
            return action
    

    def _eval_llm(
        self,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        config = self.config.clone()


        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                if config.EVAL.OVERWRITE_RESULTS:
                    print("Overwriting previous results...")
                else:
                    print(f"skipping -- evaluation exists. File path: {fname}")
                    user_input = input("Do you want to overwrite the results? (yes/no): ").strip().lower()
                    if user_input != "yes":
                        print("Skipping evaluation.")
                        return
                    else:
                        print("Overwriting previous results...")
                

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        ) 

        #envs.number_of_episodes = [1] # set the number of episodes
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval() 
        self.waypoint_predictor.eval()
        observations = envs.reset()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        # ========== Resume from checkpoint: Load completed episodes ==========
        completed_episode_ids = set()
        episode_results_file = os.path.join(
            config.RESULTS_DIR,
            f"episode_results_{config.TASK_CONFIG.DATASET.SPLIT}_r{self.local_rank}_w{self.world_size}.json"
        )
        if os.path.exists(episode_results_file):
            try:
                with open(episode_results_file, "r") as f:
                    existing_results = json.load(f)
                completed_episode_ids = set(existing_results.keys())
                # Convert to int if needed (episode_ids might be stored as strings)
                completed_episode_ids = {int(ep_id) if ep_id.isdigit() else ep_id for ep_id in completed_episode_ids}
                print(f"[Resume] Found {len(completed_episode_ids)} completed episodes: {sorted(completed_episode_ids)}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[Resume] Could not load existing results: {e}")
                completed_episode_ids = set()

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        # Update progress bar if resuming
        if pbar is not None and len(completed_episode_ids) > 0:
            pbar.update(len(completed_episode_ids))
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # set up the logger
        log_file = os.path.join(REPO_ROOT, f"navigator_{config.LOG_FILE}")
        if os.path.exists(log_file): os.remove(log_file)
        import logging
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_logger")
        nav_logger.addHandler(logging.FileHandler(filename=log_file))
        
        dataset_name = "R2R"
        cache_dir = os.path.join(REPO_ROOT, "cache_files", dataset_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        actions_cache_path = os.path.join(cache_dir, "actions_cache.json")
        if os.path.exists(actions_cache_path): 
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {}
        
        navigator = Open_Nav(self.device,config.LLM, config.API_KEY)
        current_step = 0
        current_action_idx = 0
        nav_history = []
        error_number = 0
        chosen_images = []
        chosen_images_descriptions = []  # Descriptions for each image
        env_actions_history = []  # Record environment actions for backtracking
        previous_position = None  # Track previous position to detect if stuck
        stuck_directions = set()  # Track directions that caused the agent to get stuck
        last_chosen_vp = None  # Track the last chosen viewpoint

        # Step-level statistics for current episode
        episode_step_latencies = []  # List of latencies for each step in current episode
        episode_step_input_tokens = []  # List of input tokens for each step
        episode_step_output_tokens = []  # List of output tokens for each step
        step_start_time = None  # Track start time of current step

        # Global statistics across all episodes (for computing overall average)
        global_total_latency = 0.0
        global_total_input_tokens = 0
        global_total_output_tokens = 0
        global_total_steps = 0

        # ========== Resume: Load existing global statistics ==========
        if len(completed_episode_ids) > 0:
            running_stats_file = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}_running.json"
            )
            if os.path.exists(running_stats_file):
                try:
                    with open(running_stats_file, "r") as f:
                        running_stats = json.load(f)
                    # Recover global statistics from running stats
                    if 'global_total_steps' in running_stats and running_stats['global_total_steps'] > 0:
                        global_total_steps = running_stats['global_total_steps']
                        global_total_latency = running_stats.get('global_avg_latency_per_step', 0) * global_total_steps
                        global_total_input_tokens = int(running_stats.get('global_avg_input_tokens_per_step', 0) * global_total_steps)
                        global_total_output_tokens = int(running_stats.get('global_avg_output_tokens_per_step', 0) * global_total_steps)
                        print(f"[Resume] Loaded global stats: {global_total_steps} steps, {global_total_latency:.2f}s total latency")
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"[Resume] Could not load global statistics: {e}")
        
        # Initialize debug info for visualization
        debug_info = {
            "experiment_name": config.EVAL.SPLIT,
            "episodes": []
        }
        
        # Add the initial forward-looking image (Direction 0)
        if '0' in images_list:
            chosen_images.append(images_list['0']['rgb'].copy())
            chosen_images_descriptions.append("Initial position: Agent standing at start point looking forward")
            nav_logger.info("Added initial forward-looking image to sequence")
        
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()

            # ========== Resume: Skip already completed episodes ==========
            current_ep_id = current_episodes[0].episode_id
            if current_ep_id in completed_episode_ids or str(current_ep_id) in completed_episode_ids:
                print(f"[Resume] Skipping already completed episode {current_ep_id}")
                # Mark as done and reset to next episode
                stats_episodes[current_ep_id] = None  # Placeholder to track skipped episodes
                observations[0] = envs.reset_at(0)[0]
                instruction, images_list = self.generate_input(observations[0])
                observations = extract_instruction_tokens(
                    observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)

                # Reset episode-level variables
                current_step = 0
                current_action_idx = 0
                nav_history = []
                chosen_images = []
                chosen_images_descriptions = []
                env_actions_history = []
                previous_position = None
                stuck_directions = set()
                last_chosen_vp = None
                episode_step_latencies = []
                episode_step_input_tokens = []
                episode_step_output_tokens = []

                # Add initial image for next episode
                if '0' in images_list:
                    chosen_images.append(images_list['0']['rgb'].copy())
                    chosen_images_descriptions.append("Initial position: Agent standing at start point looking forward")
                continue

            positions = []; headings = []
            for ob_i in range(len(current_episodes)):
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            # ==========Navigator start==========
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info("Instruction: "+instruction)
            
            # Collect episode info for debug.json
            # Always create a new entry for each navigation attempt to avoid mixing steps from different runs
            episode_id = current_episodes[0].episode_id
            
            # Check if this is a continuation of the current episode or a new attempt
            # If it's the first step (current_step == 0), create a new episode entry
            if current_step == 0:
                episode_info = {
                    "episode_id": episode_id,
                    "scene_id": current_episodes[0].scene_id,
                    "instruction": instruction,
                    "steps": []  # Will store step-by-step data
                }
                debug_info["episodes"].append(episode_info)
            # Otherwise, use the last episode entry (which should be the current one)
            else:
                episode_info = debug_info["episodes"][-1] if debug_info["episodes"] else None
                # Safety check: create new entry if somehow episode_info is None
                if episode_info is None:
                    episode_info = {
                        "episode_id": episode_id,
                        "scene_id": current_episodes[0].scene_id,
                        "instruction": instruction,
                        "steps": []
                    }
                    debug_info["episodes"].append(episode_info)
            actions, landmark_list = "", []
            if instruction not in actions_cache.keys():
                actions = navigator.get_actions(instruction)
                action_list = actions.split("\n")
                # Extract landmarks for each action individually
                landmark_list = []
                for i, action in enumerate(action_list):
                    if action.strip():  # Skip empty actions
                        landmarks = navigator.get_landmarks(action)
                        landmark_list.append(landmarks.replace("- ", "").split("\n"))
                    else:
                        landmark_list.append("")
                        nav_logger.info(f"Action {i} is empty, skipping landmark extraction")
                actions_cache[instruction] = {"actions": actions, "landmark_list": landmark_list}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
            else:
                actions = actions_cache[instruction]["actions"]
                landmark_list = actions_cache[instruction]["landmark_list"]

            action_list = actions.split("\n")

            # Store sub-instructions and landmarks in episode_info for the first step
            if current_step == 0:  # First step after initialization
                nav_logger.info("Sub-instructions: "+str(action_list))
                nav_logger.info("Landmarks: " + str(landmark_list))

                episode_info["sub_instructions"] = action_list
                episode_info["landmarks"] = landmark_list

            # Use MAX_EPISODE_STEPS from config instead of hardcoded values
            step_length = self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS

            stop_flag = False
            current_step += 1

            # Record step start time and reset LLM token accumulator
            step_start_time = time.time()
            navigator.llm.reset_step_tokens()

            nav_logger.info(f"-------------------- Step {current_step} --------------------")
            with torch.no_grad():
                # candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net( 
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
            
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(batch_distances[-1], batch_angles, images_list)
            if current_action_idx < len(action_list):
                nav_logger.info("Current sub-instruction: "+action_list[current_action_idx]
                                +"Current landmarks: "+str(landmark_list[current_action_idx]))

            nav_logger.info("========== Get Observation ==========")
            observation, observe_dict = navigator.observe_environment(nav_logger, current_step, images_dict)
            
            # Prepare step data for debug.json
            current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else []
            step_data = {
                "step_index": current_step,
                "viewpoints": {},
                "current_action": action_list[current_action_idx] if current_action_idx < len(action_list) else "Finishing",
                "current_landmarks": current_landmarks
            }
            
            # Store all viewpoint images with base64 encoding and mark candidates
            for vp_id, vp_data in images_dict.items():
                # Get base64 data that's already encoded in generate_input
                rgb_base64 = vp_data.get('base64') if isinstance(vp_data, dict) else None

                step_data["viewpoints"][vp_id] = {
                    "rgb_base64": rgb_base64,
                    "is_chosen": False  # Will be updated after selection
                }
            
            if len(nav_history) == 0:
                history_traj = "Step 0 start position. "

            if not stop_flag:                               
                nav_logger.info("========== Next Action Prediction ==========")
                if current_action_idx + 1 < len(action_list):
                    next_instruction = action_list[current_action_idx + 1]
                else:
                    next_instruction = 'Stop.'

                # Filter out stuck directions from available observations
                filtered_observe_dict = {k: v for k, v in observe_dict.items() if k not in stuck_directions}
                filtered_images_dict = {k: v for k, v in images_dict.items() if k not in stuck_directions}

                if len(filtered_observe_dict) == 0:
                    nav_logger.error("All directions are stuck! Using original observations.")
                    filtered_observe_dict = observe_dict
                    filtered_images_dict = images_dict
                else:
                    nav_logger.info(f"Filtered out {len(stuck_directions)} stuck directions: {stuck_directions}")

                next_vp, thought, completion_estimation, gpt_interaction = navigator.move_to_next_vp_single(nav_logger, action_list[current_action_idx], landmark_list[current_action_idx], history_traj, observation, filtered_observe_dict, filtered_images_dict, next_instruction)

                # Track the chosen viewpoint
                last_chosen_vp = next_vp

                # Update step data with chosen viewpoint and GPT interaction
                step_data["chosen_viewpoint"] = next_vp
                step_data["gpt_interaction"] = gpt_interaction
                if next_vp in step_data["viewpoints"]:
                    step_data["viewpoints"][next_vp]["is_chosen"] = True

                # Add step data to episode info
                episode_info["steps"].append(step_data)

                # Save history
                curr_observe = observe_dict[next_vp]
                nav_logger.info("========== save history ==========")
                nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)

                # Only add image if not stuck (will be determined after env.step)
                # For now, we'll add it and potentially remove it later if stuck
                chosen_images.append(images_dict[next_vp]['rgb'].copy())

                # Add description for this image
                angle_deg = np.rad2deg(radius_dict[next_vp]) if next_vp in radius_dict else 0
                direction_desc = ""
                if -15 <= angle_deg <= 15:
                    direction_desc = "forward"
                elif 15 < angle_deg <= 45:
                    direction_desc = "front-left (30°)"
                elif 45 < angle_deg <= 75:
                    direction_desc = "left (60°)"
                elif 75 < angle_deg <= 105:
                    direction_desc = "left (90°)"
                elif 105 < angle_deg <= 135:
                    direction_desc = "back-left (120°)"
                elif 135 < angle_deg <= 165:
                    direction_desc = "back-left (150°)"
                elif angle_deg > 165 or angle_deg < -165:
                    direction_desc = "backward (180°)"
                elif -165 <= angle_deg < -135:
                    direction_desc = "back-right (150°)"
                elif -135 <= angle_deg < -105:
                    direction_desc = "back-right (120°)"
                elif -105 <= angle_deg < -75:
                    direction_desc = "right (90°)"
                elif -75 <= angle_deg < -45:
                    direction_desc = "right (60°)"
                elif -45 <= angle_deg < -15:
                    direction_desc = "front-right (30°)"
                
                image_desc = f"Step {current_step}: Agent at previous position looking {direction_desc} towards chosen next viewpoint"
                chosen_images_descriptions.append(image_desc)
                nav_logger.info(f"Added image with description: {image_desc}")

                nav_logger.info("========== Review History after navigation ==========")
                history_traj = navigator.review_history(nav_logger, nav_history) if len(nav_history) > 0 else "Step 0 start position. "

                nav_logger.info("========== Estimate Completion Progress ==========")
                nav_logger.info(f"Completion estimation result: '{completion_estimation}'")
                step_data["estimation_result"] = completion_estimation

                if completion_estimation == "Yes":
                    nav_logger.info("========== Navigation Decision Agent ==========")
                    
                    # Import the decision agent
                    from vlnce_baselines.common.navigator.decision_agent import (
                        NavigationDecisionAgent, 
                        DecisionContext,
                        NavigationDecision
                    )
                    
                    # Initialize decision agent with enabled abilities from config
                    enabled_abilities = getattr(config, 'NAVIGATION_AGENT', {}).get('ENABLED_META_ABILITIES',
                                               ["continue", "stay", "backtrack", "look_around"])
                    decision_agent = NavigationDecisionAgent(navigator.llm, enabled_abilities, nav_logger)
                    
                    # Prepare context for decision
                    actions_so_far = " ".join(action_list[:current_action_idx+1])
                    current_action = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
                    current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else []
                    
                    decision_context = DecisionContext(
                        chosen_images=chosen_images,
                        current_action=current_action,
                        current_landmarks=current_landmarks,
                        actions_completed=actions_so_far,
                        history_trajectory=history_traj,
                        current_step=current_step,
                        total_actions=len(action_list),
                        current_action_idx=current_action_idx,
                        image_descriptions=chosen_images_descriptions  # Pass the descriptions
                    )
                    
                    # Get decision from agent (use informed decision for better understanding)
                    # You can switch between make_decision() and make_informed_decision()
                    use_code_analysis = getattr(config, 'USE_CODE_ANALYSIS', True)
                    if use_code_analysis:
                        decision, confidence, reasoning, decision_interaction = decision_agent.make_informed_decision_with_capture(decision_context)
                    else:
                        decision, confidence, reasoning, decision_interaction = decision_agent.make_decision_with_capture(decision_context)
                    nav_logger.info(f"Agent decision: {decision.value} (confidence: {confidence}/10)")
                    nav_logger.info(f"Agent reasoning: {reasoning}")

                    # Store decision agent interaction in step data
                    if decision_interaction:
                        step_data["decision_agent_interaction"] = decision_interaction
                        nav_logger.info("Decision agent interaction data stored in step_data")
                    else:
                        nav_logger.warning("No decision interaction data received")
                    
                    # Execute decision
                    if decision == NavigationDecision.CONTINUE:
                        # Move to next sub-instruction
                        if current_action_idx < len(action_list) - 1:
                            nav_logger.info("Agent decided to continue to next sub-instruction")
                            current_action_idx += 1
                            nav_history = []
                            history_traj = "Step 0 start position. "
                            # Clear stuck directions when moving to new instruction
                            stuck_directions.clear()
                            nav_logger.info("Cleared stuck directions for new sub-instruction")
                        else:
                            nav_logger.info("Completed all sub-instructions")
                            stop_flag = True
                    
                    elif decision == NavigationDecision.LOOK_AROUND:
                        # Explore more viewpoints for information
                        nav_logger.info("Agent decided to look around for more information")
                        current_action = action_list[current_action_idx] if current_action_idx < len(action_list) else ""
                        current_landmarks = landmark_list[current_action_idx] if current_action_idx < len(landmark_list) else ""
                        
                        # Gather enhanced observations
                        enhanced_observation, enhanced_observe_dict = navigator.look_around(
                            nav_logger, current_step, images_dict, current_action, current_landmarks, history_traj
                        )
                        
                        observation = enhanced_observation
                        observe_dict = enhanced_observe_dict
                        nav_logger.info("Enhanced observations gathered from all viewpoints")
                    
                    elif decision == NavigationDecision.BACKTRACK:
                        # Go back to previous position
                        nav_logger.info("Agent decided to backtrack")
                        if len(env_actions_history) > 0:
                            # Execute reverse action
                            last_action = env_actions_history.pop()
                            reverse_action = self._create_reverse_action(last_action)
                            nav_logger.info(f"Executing reverse action: {reverse_action}")
                            
                            outputs = envs.step([reverse_action])
                            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                            instruction, images_list = self.generate_input(observations[-1])
                            
                            # Update history
                            if len(nav_history) > 0:
                                nav_history.pop()
                            if len(chosen_images) > 1:  # Keep the initial image
                                chosen_images.pop()
                                chosen_images_descriptions.pop()
                            
                            nav_logger.info("Successfully backtracked")
                            backtrack_flag = True
                        else:
                            nav_logger.info("Cannot backtrack - no previous actions")
                    
                    elif decision == NavigationDecision.STAY:
                        # Continue with current sub-instruction
                        nav_logger.info("Agent decided to stay with current sub-instruction")
                    
                    # Check if navigation should stop
                    if decision_agent.should_stop_navigation(decision, decision_context):
                        nav_logger.info("Agent determined navigation should stop")
                        stop_flag = True

                elif completion_estimation == "No":
                    nav_logger.info("Instruction not yet completed - continuing with current instruction")
                    nav_logger.info("Skipping decision agent visualization - continuing with standard navigation")

                else:
                    nav_logger.error(f"Unexpected estimation result: {completion_estimation}")

            try:
                if not stop_flag:
                    env_actions = []
                    env_actions.append({'action':
                        {'action': 4,
                        'action_args':{
                            'angle': radius_dict[next_vp],
                            'distance': distance_dict[next_vp],
                        }}})
                    nav_logger.info(f"The final env action: {env_actions}")
                    # Record the action for potential backtracking
                    env_actions_history.append(env_actions[0])
                    outputs = envs.step(env_actions)

                    # Step completed - record step statistics
                    step_latency = time.time() - step_start_time
                    step_tokens = navigator.llm.get_step_tokens()
                    episode_step_latencies.append(step_latency)
                    episode_step_input_tokens.append(step_tokens['input_tokens'])
                    episode_step_output_tokens.append(step_tokens['output_tokens'])
                    nav_logger.info(f"Step {current_step} stats: latency={step_latency:.2f}s, input_tokens={step_tokens['input_tokens']}, output_tokens={step_tokens['output_tokens']}")

                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                    instruction, images_list = self.generate_input(observations[-1])

                    # Check if agent is stuck (position hasn't changed)
                    is_stuck = False
                    if current_step == step_length:
                        dones[0] = True
                    else:
                        for j, ob in enumerate(observations):
                            new_positions = ob.pop('positions')
                            new_collisions = ob.pop('collisions')

                            # Check if stuck: compare current position with previous position
                            if previous_position is not None and len(new_positions) > 0:
                                current_position = new_positions[-1]
                                position_diff = np.linalg.norm(np.array(current_position) - np.array(previous_position))

                                if position_diff < 0.1:  # Threshold: if moved less than 0.1 meters, consider stuck
                                    is_stuck = True
                                    nav_logger.warning(f"Agent is STUCK! Position changed by only {position_diff:.4f}m")
                                    nav_logger.warning(f"Previous position: {previous_position}")
                                    nav_logger.warning(f"Current position: {current_position}")

                                    # Add the stuck direction to the blacklist
                                    if last_chosen_vp is not None:
                                        stuck_directions.add(last_chosen_vp)
                                        nav_logger.warning(f"Added direction '{last_chosen_vp}' to stuck directions blacklist")
                                        nav_logger.warning(f"Total stuck directions: {stuck_directions}")

                                    # Remove the last added image and description since we're stuck
                                    if len(chosen_images) > 0:
                                        removed_img = chosen_images.pop()
                                        nav_logger.warning(f"Removed stuck observation image (total images now: {len(chosen_images)})")
                                    if len(chosen_images_descriptions) > 0:
                                        removed_desc = chosen_images_descriptions.pop()
                                        nav_logger.warning(f"Removed stuck image description: {removed_desc}")

                                    # Also remove from nav_history since this movement failed
                                    if len(nav_history) > 0:
                                        nav_history.pop()
                                        nav_logger.warning("Removed last navigation history entry due to stuck")
                                else:
                                    nav_logger.info(f"Agent moved {position_diff:.4f}m successfully")

                                # Update previous position
                                previous_position = current_position
                            elif len(new_positions) > 0:
                                # First movement, just record position
                                previous_position = new_positions[-1]

                            envs.call_at(j,
                                'change_current_path',
                                {'new_path': new_positions,
                                'collisions': new_collisions}
                            )
                else:
                    dones[0] = True
                
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device)
                
                for i in range(envs.num_envs):
                    
                    if not dones[i]:
                        continue
                    
                    current_step = 0
                    current_action_idx = 0
                    nav_history = []
                    chosen_images = []
                    chosen_images_descriptions = []
                    env_actions_history = []
                    backtrack_flag = False
                    previous_position = None  # Reset position tracking for new episode
                    stuck_directions = set()  # Reset stuck directions for new episode
                    last_chosen_vp = None  # Reset last chosen viewpoint for new episode

                    # Add initial image for new episode
                    if '0' in images_list:
                        chosen_images.append(images_list['0']['rgb'].copy())
                        chosen_images_descriptions.append("Initial position: Agent standing at start point looking forward")
                        nav_logger.info("Added initial forward-looking image for new episode")
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                    if 'current_path' in envs.current_episodes()[i].info.keys():
                        positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                        collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                    distance = np.array(info['position']['distance']).astype(float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                    metric['collisions'] = collisions_.mean()
                    gt_length = distance[0]
                    metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW

                    # Calculate episode-level step statistics (average per step for this episode)
                    if len(episode_step_latencies) > 0:
                        metric['avg_latency_per_step'] = sum(episode_step_latencies) / len(episode_step_latencies)
                        metric['avg_input_tokens_per_step'] = sum(episode_step_input_tokens) / len(episode_step_input_tokens)
                        metric['avg_output_tokens_per_step'] = sum(episode_step_output_tokens) / len(episode_step_output_tokens)
                        metric['total_latency'] = sum(episode_step_latencies)
                        metric['total_input_tokens'] = sum(episode_step_input_tokens)
                        metric['total_output_tokens'] = sum(episode_step_output_tokens)

                        # Update global statistics for overall average calculation
                        global_total_latency += sum(episode_step_latencies)
                        global_total_input_tokens += sum(episode_step_input_tokens)
                        global_total_output_tokens += sum(episode_step_output_tokens)
                        global_total_steps += len(episode_step_latencies)

                        nav_logger.info(f"Episode stats: avg_latency={metric['avg_latency_per_step']:.2f}s/step, "
                                       f"avg_input_tokens={metric['avg_input_tokens_per_step']:.0f}/step, "
                                       f"avg_output_tokens={metric['avg_output_tokens_per_step']:.0f}/step")
                    else:
                        metric['avg_latency_per_step'] = 0.0
                        metric['avg_input_tokens_per_step'] = 0
                        metric['avg_output_tokens_per_step'] = 0
                        metric['total_latency'] = 0.0
                        metric['total_input_tokens'] = 0
                        metric['total_output_tokens'] = 0

                    # Reset episode-level statistics for next episode
                    episode_step_latencies = []
                    episode_step_input_tokens = []
                    episode_step_output_tokens = []

                    stats_episodes[current_episodes[i].episode_id] = metric

                    # Find current episode debug info to save with results
                    current_episode_debug = None
                    episode_id_str = str(current_episodes[i].episode_id)
                    for ep_info in debug_info["episodes"]:
                        if str(ep_info.get("episode_id")) == episode_id_str:
                            current_episode_debug = ep_info
                            break

                    # Save individual episode result immediately (includes global averages)
                    self._save_episode_result(current_episodes[i].episode_id, metric, config, current_episode_debug,
                                             global_total_latency, global_total_input_tokens,
                                             global_total_output_tokens, global_total_steps)

                    observations[i] = envs.reset_at(i)[0]
                    instruction, images_list = self.generate_input(observations[i])
                    
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)   
                
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,  
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                    rgb_frames,
                )
                headings = headings.tolist()
            except Exception as e:
                nav_logger.info(f"Error in next action prediction: {e}")
                current_step -= 1
        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)
        
        # Save debug.json with episode information
        debug_fname = os.path.join(
            config.RESULTS_DIR,
            "debug.json"
        )
        with open(debug_fname, "w") as f:
            json.dump(debug_info, f, indent=4)
        nav_logger.info(f"Saved debug info to {debug_fname}")

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        return trajectories

    def _save_episode_result(self, episode_id, metric, config, debug_episode_info=None,
                             global_total_latency=0.0, global_total_input_tokens=0,
                             global_total_output_tokens=0, global_total_steps=0):
        """Save individual episode result to JSON file immediately after evaluation."""
        if not config.EVAL.SAVE_RESULTS:
            return

        # Create results directory if it doesn't exist
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        # Path for individual episode results
        episode_results_file = os.path.join(
            config.RESULTS_DIR,
            f"episode_results_{config.TASK_CONFIG.DATASET.SPLIT}_r{self.local_rank}_w{self.world_size}.json"
        )

        # Load existing results or initialize empty dict
        if os.path.exists(episode_results_file):
            try:
                with open(episode_results_file, "r") as f:
                    episode_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                episode_results = {}
        else:
            episode_results = {}

        # Add current episode result
        episode_results[episode_id] = metric

        # Save updated results
        with open(episode_results_file, "w") as f:
            json.dump(episode_results, f, indent=4)

        # Also update aggregated stats file incrementally (with global step statistics)
        self._update_aggregated_stats(episode_results, config,
                                      global_total_latency, global_total_input_tokens,
                                      global_total_output_tokens, global_total_steps)

        # Update debug.json with episode information if provided
        if debug_episode_info is not None:
            self._update_debug_json(debug_episode_info, config)

    def _update_aggregated_stats(self, episode_results, config,
                                 global_total_latency=0.0, global_total_input_tokens=0,
                                 global_total_output_tokens=0, global_total_steps=0):
        """Update aggregated statistics file with current episode results."""
        if not config.EVAL.SAVE_RESULTS:
            return

        split = config.TASK_CONFIG.DATASET.SPLIT
        aggregated_file = os.path.join(
            config.RESULTS_DIR,
            f"stats_ckpt_{split}_running.json"
        )

        # Calculate aggregated stats from all episodes so far
        num_episodes = len(episode_results)
        if num_episodes == 0:
            return

        # Helper function to check if a value is valid (not NaN or Infinity)
        def is_valid_number(x):
            return x is not None and not math.isnan(x) and not math.isinf(x)

        # Keys to exclude from episode-level averaging (step-level stats handled separately)
        step_level_keys = {'avg_latency_per_step', 'avg_input_tokens_per_step', 'avg_output_tokens_per_step',
                          'total_latency', 'total_input_tokens', 'total_output_tokens'}

        aggregated_stats = {}
        for stat_key in next(iter(episode_results.values())).keys():
            if stat_key in step_level_keys:
                continue
            # Filter out invalid values (NaN, Infinity)
            valid_values = [v[stat_key] for v in episode_results.values() if is_valid_number(v[stat_key])]
            if len(valid_values) > 0:
                aggregated_stats[stat_key] = sum(valid_values) / len(valid_values)
            else:
                aggregated_stats[stat_key] = 0.0  # Default to 0 if no valid values

        # Add episode count
        aggregated_stats['episodes_evaluated'] = num_episodes

        # Add global step-level statistics (average across ALL steps from ALL episodes)
        if global_total_steps > 0:
            aggregated_stats['global_avg_latency_per_step'] = global_total_latency / global_total_steps
            aggregated_stats['global_avg_input_tokens_per_step'] = global_total_input_tokens / global_total_steps
            aggregated_stats['global_avg_output_tokens_per_step'] = global_total_output_tokens / global_total_steps
            aggregated_stats['global_total_steps'] = global_total_steps

        # Save aggregated stats
        with open(aggregated_file, "w") as f:
            json.dump(aggregated_stats, f, indent=4)

    def _update_debug_json(self, debug_episode_info, config):
        """Update debug.json file with episode information incrementally."""
        if not config.EVAL.SAVE_RESULTS:
            return

        debug_file = os.path.join(
            config.RESULTS_DIR,
            "debug.json"
        )

        # Load existing debug data or initialize
        if os.path.exists(debug_file):
            try:
                with open(debug_file, "r") as f:
                    debug_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                debug_data = {
                    "experiment_name": config.EVAL.SPLIT,
                    "episodes": []
                }
        else:
            debug_data = {
                "experiment_name": config.EVAL.SPLIT,
                "episodes": []
            }

        # Check if this episode already exists (update) or is new (append)
        episode_id = debug_episode_info.get("episode_id")
        existing_episode_idx = None

        for idx, episode in enumerate(debug_data["episodes"]):
            if episode.get("episode_id") == episode_id:
                existing_episode_idx = idx
                break

        if existing_episode_idx is not None:
            # Update existing episode
            debug_data["episodes"][existing_episode_idx] = debug_episode_info
        else:
            # Add new episode
            debug_data["episodes"].append(debug_episode_info)

        # Save updated debug data
        with open(debug_file, "w") as f:
            json.dump(debug_data, f, indent=4)

    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_llm()

