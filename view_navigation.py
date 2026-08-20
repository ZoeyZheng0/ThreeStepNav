# streamlit run view_navigation.py
import streamlit as st
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timezone
import cv2
from PIL import Image, ImageDraw
import os

# Configuration
st.set_page_config(
    page_title="Open-Nav Visualization Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main folders (override with the OPENNAV_LOGS_DIR environment variable)
LOGS_DIR = Path(os.environ.get("OPENNAV_LOGS_DIR", Path(__file__).resolve().parent / "logs"))
EVAL_RESULTS_DIR = LOGS_DIR / "eval_results"

def load_experiment_data(experiment_name):
    """Load experiment debug data and metrics"""
    experiment_data = {}
    
    # Load debug.json
    debug_path = EVAL_RESULTS_DIR / experiment_name / "debug.json"
    if debug_path.exists():
        with open(debug_path, 'r') as f:
            experiment_data['debug'] = json.load(f)
    
    # Load stats json file - check both original and running stats files
    stats_files = list((EVAL_RESULTS_DIR / experiment_name).glob("stats_ep_ckpt_*.json"))
    running_stats_files = list((EVAL_RESULTS_DIR / experiment_name).glob("stats_ckpt_*_running.json"))
    episode_results_files = list((EVAL_RESULTS_DIR / experiment_name).glob("episode_results_*.json"))

    if stats_files:
        # Use the original stats file (final results)
        with open(stats_files[0], 'r') as f:
            experiment_data['stats'] = json.load(f)
    elif running_stats_files:
        # Fallback to running stats file (incremental results)
        with open(running_stats_files[0], 'r') as f:
            experiment_data['stats'] = json.load(f)

    # Load individual episode results if available
    if episode_results_files:
        with open(episode_results_files[0], 'r') as f:
            experiment_data['episode_results'] = json.load(f)
    
    return experiment_data


def display_gpt_interaction(step_data):
    """Display GPT input and output for a step"""
    gpt_data = step_data.get('gpt_interaction', {})

    if gpt_data:

        # # Show response metadata first if available
        # if 'metadata' in gpt_data:
        #     metadata = gpt_data['metadata']
        #     col1, col2, col3, col4 = st.columns(4)
        #     with col1:
        #         if 'model' in metadata:
        #             st.metric("Model", metadata['model'])
        #     with col2:
        #         if 'num_images' in metadata:
        #             st.metric("Images", metadata['num_images'])
        #     with col3:
        #         if 'retry' in metadata:
        #             st.metric("Retry", "Yes" if metadata['retry'] else "No")
        #     with col4:
        #         if 'random_fallback' in metadata:
        #             st.metric("Fallback", "Yes" if metadata['random_fallback'] else "No")

        # Display all prompts and response in full width without tabs
        system_prompt = gpt_data.get('system_prompt', 'No system prompt recorded')
        st.text_area("System Prompt Content", system_prompt, height=500)

        user_prompt = gpt_data.get('user_prompt', 'No user prompt recorded')
        st.text_area("User Prompt Content", user_prompt, height=300)

        gpt_response = gpt_data.get('response', 'No response recorded')
        st.text_area("GPT Response Content", gpt_response, height=250)


def display_decision_agent_interaction(step_data):
    """Display Decision Agent GPT input and output for a step"""
    decision_data = step_data.get('decision_agent_interaction', {})

    if decision_data:
        st.markdown("#### Step 3: Look backward (global view)")

        # Show response metadata first if available
        if 'metadata' in decision_data:
            metadata = decision_data['metadata']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'model' in metadata:
                    st.metric("Decision Model", metadata['model'])
            with col2:
                if 'num_images' in metadata:
                    st.metric("Images Analyzed", metadata['num_images'])
            with col3:
                if 'method' in metadata:
                    method_display = "Informed" if metadata['method'] == 'make_informed_decision' else "Standard"
                    st.metric("Method", method_display)
            with col4:
                if 'code_analysis_used' in metadata:
                    st.metric("Code Analysis", "Yes" if metadata['code_analysis_used'] else "No")

        # Display images sent to decision agent if available
        images_base64 = decision_data.get('images_base64', {})
        if images_base64:
            st.markdown("**Images Sent to Decision Agent:**")
            # Organize images in rows
            cols_per_row = 4
            image_ids = sorted(images_base64.keys(), key=lambda x: int(x))

            for i in range(0, len(image_ids), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(image_ids):
                        img_id = image_ids[i + j]
                        img_base64 = images_base64[img_id]

                        with col:
                            st.image(img_base64, caption=f"Decision Image {img_id}", use_container_width=True)

        # Display all prompts and response in full width without tabs
        st.markdown("**Decision Agent System Prompt:**")
        system_prompt = decision_data.get('system_prompt', 'No system prompt recorded')
        st.text_area("Decision Agent System Prompt Content", system_prompt, height=200)

        st.markdown("**Decision Agent User Prompt:**")
        user_prompt = decision_data.get('user_prompt', 'No user prompt recorded')
        st.text_area("Decision Agent User Prompt Content", user_prompt, height=400)

        st.markdown("**Decision Agent Response:**")
        decision_response = decision_data.get('response', 'No response recorded')
        st.text_area("Decision Agent Response Content", decision_response, height=300)
    else:
        # Show when decision agent was not called
        st.markdown("#### 🧠 Decision Agent Interaction")
        st.info("Decision agent interaction data not available for this step")


def display_navigation_steps(episode_info, episode_id):

    steps = episode_info.get('steps', [])
    if not steps or len(steps) == 0:
        st.warning("No step data available")
        return

    # Step selector - handle single step case
    if len(steps) == 1:
        step_index = 0
        st.info("Only one step available")
    else:
        step_display = st.slider("Select Navigation Step Index", 1, len(steps), 1)
        step_index = step_display - 1  # Convert from 1-based to 0-based indexing
    step_data = steps[step_index]

    # Display step information with sub-instructions and landmarks
    current_action = step_data.get('current_action', 'N/A')
    current_landmarks = step_data.get('current_landmarks', [])

    # Format landmarks display
    landmarks_text = ""
    if current_landmarks:
        if isinstance(current_landmarks, list):
            landmarks_text = " | ".join([landmark.strip() for landmark in current_landmarks if landmark.strip()])
        else:
            landmarks_text = str(current_landmarks)

    if landmarks_text:
        st.markdown(f"**Current sub-instruction and landmarks**: {current_action} | {landmarks_text}")
    else:
        st.markdown(f"**Current sub-instruction and landmarks**: {current_action} (No landmarks specified)")

    # Display viewpoints first
    viewpoints = step_data.get('viewpoints', {})
    if viewpoints:
        st.markdown("##### Input candidate viewpoints")

        # Organize viewpoints in rows
        cols_per_row = 4
        viewpoint_ids = sorted(viewpoints.keys())

        for i in range(0, len(viewpoint_ids), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(viewpoint_ids):
                    vp_id = viewpoint_ids[i + j]
                    vp_data = viewpoints[vp_id]

                    with col:
                        # Determine label with checkmark for chosen viewpoint
                        if vp_data.get('is_chosen', False):
                            label = f"VP {vp_id} ✅"
                        else:
                            label = f"VP {vp_id}"

                        # Display RGB image from base64 or file path
                        if 'rgb_base64' in vp_data and vp_data['rgb_base64']:
                            # Use base64 image if available
                            st.image(vp_data['rgb_base64'], caption=label, use_container_width=True)
                        else:
                            # Fallback to file path method
                            rgb_path = Path(EVAL_RESULTS_DIR) / "episode_images" / str(episode_id) / vp_data['rgb_path']
                            try:
                                st.image(str(rgb_path), caption=label, use_container_width=True)
                            except:
                                st.error(f"Image not found: {vp_data['rgb_path']}")

    display_gpt_interaction(step_data)

    # Display Decision Agent interaction
    display_decision_agent_interaction(step_data)
    

def display_episode_info(debug_data, stats_data, selected_episode_id):
    """Display information for a selected episode"""
    
    # Find the selected episode in debug data
    episode_info = None
    for ep in debug_data.get('episodes', []):
        if ep['episode_id'] == selected_episode_id:
            episode_info = ep
            break
    
    if episode_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Episode Information**")
            st.info(f"Episode ID: {episode_info['episode_id']}")
            # Extract just the scene name from the full path
            scene_name = episode_info['scene_id'].split('/')[-1].replace('.glb', '')
            st.info(f"Scene ID: {scene_name}")
        
        with col2:
            st.text_area("Overall Instruction", episode_info['instruction'], height=100)
        
        # Display metrics if available
        if stats_data and ('episodes_evaluated' not in stats_data) and selected_episode_id in stats_data:
            st.markdown("### 📊 Episode Metrics")
            metrics = stats_data[selected_episode_id]
            
            # Create metric columns
            metric_cols = st.columns(7)
            
            with metric_cols[0]:
                st.metric("Success", "✅" if metrics.get('success', 0) == 1 else "❌")
            
            with metric_cols[1]:
                st.metric("Oracle Success", "✅" if metrics.get('oracle_success', 0) == 1 else "❌")
            
            with metric_cols[2]:
                st.metric("SPL", f"{metrics.get('spl', 0):.3f}")
            
            with metric_cols[3]:
                st.metric("NDTW", f"{metrics.get('ndtw', 0):.3f}")
            
            with metric_cols[4]:
                st.metric("Steps", f"{metrics.get('steps_taken', 0):.0f}")
            
            with metric_cols[5]:
                st.metric("Distance to Goal", f"{metrics.get('distance_to_goal', 0):.2f}m")
            
            with metric_cols[6]:
                st.metric("Path Length", f"{metrics.get('path_length', 0):.2f}m")
            
            # Detailed metrics table
            with st.expander("View All Metrics"):
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df, use_container_width=True)
        
        # Display sub-instructions with landmarks
        st.markdown("### Step 1: Look forward (global view)")
        if 'sub_instructions' in episode_info and 'landmarks' in episode_info:
            sub_instructions = episode_info['sub_instructions']
            landmarks = episode_info['landmarks']

            if isinstance(sub_instructions, list) and isinstance(landmarks, list):
                for i, instruction in enumerate(sub_instructions):
                    st.markdown(f"{instruction} {landmarks[i]}")
            else:
                st.warning("Sub-instructions or landmarks data not available or in unexpected format")
        else:
            st.warning("Sub-instructions and landmarks data not found in episode info")

        # Display step-by-step navigation if available
        st.markdown("### Step 2: Look now (local view)")
        if 'steps' in episode_info and episode_info['steps']:
            display_navigation_steps(episode_info, selected_episode_id)

def display_experiment_summary(debug_data, stats_data, episode_results_data=None, selected_episode_id=None):
    """Display overall experiment summary"""
    st.markdown("#### Experiment Summary")

    if stats_data:
        # Check if stats_data is aggregated (single dict) or individual episodes (dict of dicts)
        if 'episodes_evaluated' in stats_data:
            # This is aggregated running stats
            total_episodes = stats_data.get('episodes_evaluated', 0)
            successful_episodes = int(stats_data.get('success', 0) * total_episodes)
            oracle_successful_episodes = int(stats_data.get('oracle_success', 0) * total_episodes)
            avg_spl = stats_data.get('spl', 0)
            avg_ndtw = stats_data.get('ndtw', 0)
            avg_steps = stats_data.get('steps_taken', 0)
            avg_distance = stats_data.get('distance_to_goal', 0)
        else:
            # This is individual episode data
            total_episodes = len(stats_data)
            successful_episodes = sum(1 for m in stats_data.values() if m.get('success', 0) == 1)
            oracle_successful_episodes = sum(1 for m in stats_data.values() if m.get('oracle_success', 0) == 1)
            avg_spl = np.mean([m.get('spl', 0) for m in stats_data.values()])
            avg_ndtw = np.mean([m.get('ndtw', 0) for m in stats_data.values()])
            avg_steps = np.mean([m.get('steps_taken', 0) for m in stats_data.values()])
            avg_distance = np.mean([m.get('distance_to_goal', 0) for m in stats_data.values()])
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Episodes", total_episodes)
            st.metric("Success Rate", f"{(successful_episodes/total_episodes)*100:.1f}%")
            st.metric("Oracle Success Rate", f"{(oracle_successful_episodes/total_episodes)*100:.1f}%")
        
        with col2:
            st.metric("Avg SPL", f"{avg_spl:.3f}")
            st.metric("Avg NDTW", f"{avg_ndtw:.3f}")
        
        with col3:
            st.metric("Avg Steps", f"{avg_steps:.1f}")
            st.metric("Avg Distance to Goal", f"{avg_distance:.2f}m")
        
        with col4:
            # Success distribution pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Success', 'Failure'],
                values=[successful_episodes, total_episodes - successful_episodes],
                hole=0.3,
                marker_colors=['#00cc44', '#ff4444']
            )])
            fig.update_layout(
                title="Success Distribution",
                height=200,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Episode metrics visualization - only for individual episode data
        if 'episodes_evaluated' not in stats_data:
            st.markdown("### 📊 Episode Metrics Distribution")

            # Create dataframe for visualization
            episodes_list = []
            for ep_id, metrics in stats_data.items():
                episodes_list.append({
                    'Episode ID': ep_id,
                    'Success': metrics.get('success', 0),
                    'Oracle Success': metrics.get('oracle_success', 0),
                    'SPL': metrics.get('spl', 0),
                    'NDTW': metrics.get('ndtw', 0),
                    'Steps': metrics.get('steps_taken', 0),
                    'Distance to Goal': metrics.get('distance_to_goal', 0),
                    'Path Length': metrics.get('path_length', 0)
                })

            episodes_df = pd.DataFrame(episodes_list)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # SPL distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=episodes_df['SPL'],
                    nbinsx=20,
                    name='SPL Distribution',
                    marker_color='blue'
                ))
                fig.update_layout(
                    title="SPL Distribution",
                    xaxis_title="SPL",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distance to goal distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=episodes_df['Distance to Goal'],
                    nbinsx=20,
                    name='Distance Distribution',
                    marker_color='green'
                ))
                fig.update_layout(
                    title="Distance to Goal Distribution",
                    xaxis_title="Distance (m)",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            # Full episode table
            with st.expander("View All Episodes"):
                st.dataframe(
                    episodes_df.style.format({
                        'Success': lambda x: '✅' if x == 1 else '❌',
                        'Oracle Success': lambda x: '✅' if x == 1 else '❌',
                        'SPL': '{:.3f}',
                        'NDTW': '{:.3f}',
                        'Steps': '{:.0f}',
                        'Distance to Goal': '{:.2f}',
                        'Path Length': '{:.2f}'
                    }),
                    use_container_width=True
                )
        else:
            if debug_data and 'episodes' in debug_data and selected_episode_id is not None:
                episodes = debug_data['episodes']
                episode_ids = [ep.get('episode_id') for ep in episodes]

                if selected_episode_id in episode_ids:
                    # Try to get metrics from episode_results first, fallback to debug data
                    episode_metrics = None
                    episode_results = episode_results_data or {}

                    if str(selected_episode_id) in episode_results:
                        episode_metrics = episode_results[str(selected_episode_id)]

                    if episode_metrics:
                        st.markdown(f"#### Episode {selected_episode_id}")

                        # Display metrics in the same layout as experiment summary
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            success = episode_metrics.get('success', 0)
                            oracle_success = episode_metrics.get('oracle_success', 0)
                            st.metric("Success", "✅ Yes" if success == 1 else "❌ No")
                            st.metric("Oracle Success", "✅ Yes" if oracle_success == 1 else "❌ No")

                        with col2:
                            spl = episode_metrics.get('spl', 0)
                            ndtw = episode_metrics.get('ndtw', 0)
                            st.metric("SPL", f"{spl:.3f}")
                            st.metric("NDTW", f"{ndtw:.3f}")

                        with col3:
                            steps_taken = episode_metrics.get('steps_taken', 0)
                            distance_to_goal = episode_metrics.get('distance_to_goal', 0)
                            st.metric("Steps Taken", f"{steps_taken:.0f}")
                            st.metric("Distance to Goal", f"{distance_to_goal:.2f}m")

                        with col4:
                            path_length = episode_metrics.get('path_length', 0)
                            collisions = episode_metrics.get('collisions', 0)
                            st.metric("Path Length", f"{path_length:.2f}m")
                            st.metric("Collisions", f"{collisions:.3f}")

                        # Episode details table
                        with st.expander("📋 Detailed Metrics Table"):
                            metrics_df = pd.DataFrame([episode_metrics])
                            st.dataframe(
                                metrics_df.style.format({
                                    'success': lambda x: '✅ Success' if x == 1 else '❌ Failure',
                                    'oracle_success': lambda x: '✅ Success' if x == 1 else '❌ Failure',
                                    'spl': '{:.4f}',
                                    'ndtw': '{:.4f}',
                                    'steps_taken': '{:.0f}',
                                    'distance_to_goal': '{:.3f}',
                                    'path_length': '{:.3f}',
                                    'collisions': '{:.3f}'
                                }),
                                use_container_width=True
                            )
                    else:
                        st.warning(f"⚠️ Metrics not available for Episode {selected_episode_id}")
                        st.info("Episode metrics will be available after the episode completes evaluation.")
                else:
                    st.warning("No episodes found in debug data")
            else:
                st.info("💡 Individual episode metrics will be available when evaluation completes with detailed results.")
    else:
        # Show basic info when only debug data is available
        st.info("📊 **Evaluation Status**: In Progress or Stats Not Available")

        if debug_data and 'episodes' in debug_data:
            episodes = debug_data['episodes']
            total_episodes = len(episodes)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Episodes in Debug Data", total_episodes)

                # Show episode completion status
                completed_steps = sum(len(ep.get('steps', [])) for ep in episodes)
                st.metric("Total Steps Recorded", completed_steps)

            with col2:
                # Show experiment info
                exp_name = debug_data.get('experiment_name', 'Unknown')
                st.metric("Experiment Name", exp_name)

                # Show last episode info if available
                if episodes:
                    last_ep = episodes[-1]
                    last_ep_id = last_ep.get('episode_id', 'Unknown')
                    st.metric("Latest Episode ID", last_ep_id)

            st.warning("💡 **Note**: Full metrics will be available when evaluation completes or use running stats.")
        else:
            st.error("No debug data or episodes found")

def main():
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Select experiment
        st.subheader("Experiment Selection")
        # Get experiments sorted by modification time (latest first)
        experiment_dirs = [d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir()]
        if experiment_dirs:
            # Sort by modification time, latest first
            experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            experiments = [d.name for d in experiment_dirs]
            
            # Create display names with (Latest) indicator
            display_names = [f"{exp} (Latest)" if i == 0 else exp for i, exp in enumerate(experiments)]
            
            # Default to the latest experiment
            selected_display = st.selectbox("Select Experiment", display_names, index=0)
            
            # Extract actual experiment name (remove " (Latest)" if present)
            experiment = selected_display.replace(" (Latest)", "")
            
            # Load experiment data
            experiment_data = load_experiment_data(experiment)
            
            if experiment_data.get('debug'):
                st.success(f"Loaded experiment: {experiment}")
                
                # Show experiment timestamp
                exp_path = EVAL_RESULTS_DIR / experiment
                if exp_path.exists():
                    mod_time = datetime.fromtimestamp(exp_path.stat().st_mtime)
                    st.caption(f"Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Episode selection
                episodes = experiment_data['debug'].get('episodes', [])
                if episodes:
                    st.subheader("Episode Selection")
                    episode_ids = [ep['episode_id'] for ep in episodes]
                    selected_episode = st.selectbox("Select Episode", episode_ids)
                    
                    # # Display experiment info
                    # st.markdown("---")
                    # st.markdown("**Experiment Info**")
                    # st.info(f"Total Episodes: {len(episodes)}")
                    
                    if experiment_data.get('stats'):
                        stats_data = experiment_data['stats']

                        # Check if stats_data is aggregated (single dict) or individual episodes (dict of dicts)
                        if 'episodes_evaluated' in stats_data:
                            # This is aggregated running stats
                            success_rate = stats_data.get('success', 0) * 100
                            oracle_success_rate = stats_data.get('oracle_success', 0) * 100
                            episodes_eval = stats_data.get('episodes_evaluated', 0)
                            st.info(f"Episodes Evaluated: {episodes_eval}")
                            st.info(f"Success Rate: {success_rate:.1f}%")
                            st.info(f"Oracle Success Rate: {oracle_success_rate:.1f}%")
                        else:
                            # This is individual episode data
                            success_count = sum(1 for m in stats_data.values()
                                              if m.get('success', 0) == 1)
                            oracle_success_count = sum(1 for m in stats_data.values()
                                              if m.get('oracle_success', 0) == 1)
                            st.info(f"Success Rate: {(success_count/len(episodes))*100:.1f}%")
                            st.info(f"Oracle Success Rate: {(oracle_success_count/len(episodes))*100:.1f}%")
                else:
                    st.warning("No episodes found in debug.json")
            else:
                st.warning(f"No debug.json found for {experiment}")
        else:
            st.warning("No experiments found in eval_results directory")
    
    # Main content area
    if 'experiment_data' in locals() and experiment_data:
        tab1, tab2 = st.tabs(["Episode View", "Experiment Summary"])
        
        with tab1:
            if 'selected_episode' in locals():
                display_episode_info(
                    experiment_data.get('debug', {}),
                    experiment_data.get('stats', {}),
                    selected_episode
                )
        
        with tab2:
            display_experiment_summary(
                experiment_data.get('debug', {}),
                experiment_data.get('stats', {}),
                experiment_data.get('episode_results', {}),
                selected_episode
            )
    else:
        st.info("Select an experiment from the sidebar to begin visualization")

if __name__ == "__main__":
    main()