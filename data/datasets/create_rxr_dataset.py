#!/usr/bin/env python3
"""
Script to randomly select 100 English episodes from RXR dataset and save in R2R format.

This script:
1. Loads the RXR dataset (val_unseen_guide.json and val_unseen_guide_gt.json)
2. Filters for English-only episodes (en-US or en-IN)
3. Randomly selects 100 episodes
4. Converts them to R2R format
5. Saves to RXR_VLNCE_v0_preprocessed directory as .json.gz
"""

import argparse
import json
import random
import os
import gzip
from pathlib import Path

# Repository root, used for default paths
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(filepath):
    """Load JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, filepath):
    """Save data to JSON file."""
    print(f"Saving to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filepath}")


def save_json_gz(data, filepath):
    """Save data to compressed JSON.gz file."""
    print(f"Saving to {filepath}...")
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filepath}")


def filter_english_episodes(rxr_data):
    """
    Filter episodes that have English language (en-US or en-IN).

    Args:
        rxr_data: Dictionary with 'episodes' key containing list of episodes

    Returns:
        List of English episodes with their indices
    """
    english_episodes = []
    episodes = rxr_data['episodes']

    for idx, episode in enumerate(episodes):
        language = episode['instruction'].get('language', '')
        if language in ['en-US', 'en-IN']:
            english_episodes.append((idx, episode))

    return english_episodes


def convert_rxr_to_r2r_format(rxr_episode, rxr_gt_episode, episode_id):
    """
    Convert RXR episode format to R2R format.

    Args:
        rxr_episode: Episode from val_unseen_guide.json
        rxr_gt_episode: Corresponding episode from val_unseen_guide_gt.json
        episode_id: New episode ID for R2R format

    Returns:
        Dictionary in R2R format
    """
    # Calculate geodesic distance from reference path
    reference_path = rxr_episode['reference_path']
    geodesic_distance = 0.0
    for i in range(len(reference_path) - 1):
        p1 = reference_path[i]
        p2 = reference_path[i + 1]
        dist = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
        geodesic_distance += dist

    r2r_episode = {
        'episode_id': episode_id,
        'goals': rxr_episode['goals'],
        'info': {
            'geodesic_distance': geodesic_distance
        },
        'instruction': {
            'instruction_text': rxr_episode['instruction']['instruction_text'],
            'instruction_tokens': rxr_episode['instruction']['instruction_tokens']
        },
        'reference_path': rxr_episode['reference_path'],
        'scene_id': rxr_episode['scene_id'],
        'start_position': rxr_episode['start_position'],
        'start_rotation': rxr_episode['start_rotation'],
        'trajectory_id': rxr_episode['trajectory_id']
    }

    return r2r_episode


def select_random_episodes(episodes_with_distances, num_episodes=100):
    """
    Randomly select episodes from the available pool, ensuring unique trajectory IDs.

    Args:
        episodes_with_distances: List of ((idx, episode), distance) tuples
        num_episodes: Number of episodes to select (default 100)

    Returns:
        List of selected episodes
    """
    print(f"\nTotal available English episodes: {len(episodes_with_distances)}")

    # Group by trajectory_id to find unique trajectories
    trajectory_groups = {}
    for (idx, episode), distance in episodes_with_distances:
        traj_id = episode['trajectory_id']
        if traj_id not in trajectory_groups:
            trajectory_groups[traj_id] = []
        trajectory_groups[traj_id].append(((idx, episode), distance))

    print(f"Unique trajectory IDs: {len(trajectory_groups)}")

    # For each trajectory, randomly pick one episode (in case multiple instructions exist)
    random.seed(42)
    unique_episodes = []
    for traj_id, episodes in trajectory_groups.items():
        # Randomly select one episode from this trajectory
        chosen = random.choice(episodes)
        unique_episodes.append(chosen)

    # Check if we have enough unique trajectories
    if len(unique_episodes) < num_episodes:
        print(f"  WARNING: Only {len(unique_episodes)} unique trajectories available, wanted {num_episodes}")
        num_episodes = len(unique_episodes)

    # Randomly sample from unique trajectories
    selected = random.sample(unique_episodes, num_episodes)

    print(f"Randomly selected {len(selected)} episodes with unique trajectory IDs")

    # Return only episodes (not distances)
    return [ep for ep, _ in selected]


def main():
    # Paths
    parser = argparse.ArgumentParser(
        description="Randomly select English episodes from the RxR dataset and save them in R2R format."
    )
    parser.add_argument(
        '--rxr-guide', required=True,
        help="Path to the RxR val_unseen_guide.json file",
    )
    parser.add_argument(
        '--rxr-gt', required=True,
        help="Path to the RxR val_unseen_guide_gt.json file",
    )
    parser.add_argument(
        '--output-dir',
        default=str(REPO_ROOT / 'data/datasets/RXR_VLNCE_v0_preprocessed/val_unseen'),
        help="Directory to write the converted dataset into",
    )
    parser.add_argument(
        '--r2r-vocab',
        default=str(REPO_ROOT / 'data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz'),
        help="Path to an R2R .json.gz dataset providing the instruction vocab",
    )
    args = parser.parse_args()

    rxr_guide_path = args.rxr_guide
    rxr_gt_path = args.rxr_gt

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_episodes_path = os.path.join(output_dir, 'OpenNav_RXR-CE_100_bertidx.json.gz')
    output_gt_path = os.path.join(output_dir, 'val_unseen_gt.json.gz')

    # Load RXR datasets
    rxr_guide = load_json(rxr_guide_path)
    rxr_gt = load_json(rxr_gt_path)

    # Filter English episodes
    print("\nFiltering English episodes (en-US and en-IN)...")
    english_episodes = filter_english_episodes(rxr_guide)
    print(f"Found {len(english_episodes)} English episodes")

    # Calculate geodesic distance for each episode
    print("\nCalculating geodesic distances...")
    episodes_with_distances = []
    for idx, episode in english_episodes:
        reference_path = episode['reference_path']
        geodesic_distance = 0.0
        for i in range(len(reference_path) - 1):
            p1 = reference_path[i]
            p2 = reference_path[i + 1]
            dist = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            geodesic_distance += dist
        episodes_with_distances.append(((idx, episode), geodesic_distance))

    # Randomly select 100 episodes
    selected = select_random_episodes(episodes_with_distances, num_episodes=100)

    # Load R2R instruction vocab
    print("\nLoading R2R instruction vocab...")
    r2r_path = args.r2r_vocab
    with gzip.open(r2r_path, 'rt') as f:
        r2r_data = json.load(f)
        instruction_vocab = r2r_data['instruction_vocab']

    # Convert to R2R format
    print("\nConverting to R2R format...")
    r2r_episodes = []
    r2r_gt_dict = {}

    for new_id, (original_idx, rxr_episode) in enumerate(selected):
        # Get corresponding episode_id from RXR
        rxr_episode_id = rxr_episode['episode_id']

        # Get GT data
        if rxr_episode_id in rxr_gt:
            rxr_gt_episode = rxr_gt[rxr_episode_id]

            # Add GT data with new episode_id
            r2r_gt_dict[str(new_id)] = {
                'actions': rxr_gt_episode['actions'],
                'locations': rxr_gt_episode['locations'],
                'forward_steps': rxr_gt_episode['forward_steps']
            }
        else:
            print(f"WARNING: No GT data found for episode {rxr_episode_id}")
            rxr_gt_episode = None

        r2r_episode = convert_rxr_to_r2r_format(rxr_episode, rxr_gt_episode, new_id)
        r2r_episodes.append(r2r_episode)

        if (new_id + 1) % 10 == 0:
            print(f"Converted {new_id + 1}/{len(selected)} episodes")

    # Create final R2R dataset structure (same format as R2R)
    r2r_dataset = {
        'episodes': r2r_episodes,
        'instruction_vocab': instruction_vocab
    }

    # Save episode dataset as .json.gz
    save_json_gz(r2r_dataset, output_episodes_path)

    # Save GT dataset as .json.gz
    save_json_gz(r2r_gt_dict, output_gt_path)

    # Print statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total RXR episodes: {len(rxr_guide['episodes'])}")
    print(f"English episodes: {len(english_episodes)}")
    print(f"Selected episodes: {len(selected)}")
    print(f"\nOutput files:")
    print(f"  Episodes: {output_episodes_path}")
    print(f"  GT:       {output_gt_path}")

    # Distance statistics
    import numpy as np
    distances = [ep['info']['geodesic_distance'] for ep in r2r_episodes]
    print(f"\nDistance statistics:")
    print(f"  Mean:   {np.mean(distances):.2f}m")
    print(f"  Median: {np.median(distances):.2f}m")
    print(f"  Min:    {np.min(distances):.2f}m")
    print(f"  Max:    {np.max(distances):.2f}m")
    print(f"  Std:    {np.std(distances):.2f}m")

    print(f"\nLanguage distribution in selected episodes:")
    lang_count = {}
    for _, (_, ep) in enumerate(selected):
        lang = ep['instruction']['language']
        lang_count[lang] = lang_count.get(lang, 0) + 1
    for lang, count in sorted(lang_count.items()):
        print(f"  {lang}: {count} episodes")
    print("="*50)


if __name__ == '__main__':
    main()
