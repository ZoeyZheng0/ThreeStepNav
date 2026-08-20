import re
import random
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *

class Open_Nav():
    def __init__(self, device, llm_type, api_key):
        self.device = device
        self.llm = llmClient(llm_type, api_key)
        self.spatial = spatialClient(self.device)
        
    # =====================================
    # ===== Instruction Comprehension =====
    # =====================================
    def get_actions(self, instruction):
        return self.llm.gpt_infer(ACTION_DETECTION['system'], ACTION_DETECTION['user'].format(instruction))

    def get_landmarks(self, actions):
        actions = actions.replace("\n", " ")
        return self.llm.gpt_infer(LANDMARK_DETECTION['system'], LANDMARK_DETECTION['user'].format(actions))
    
    # =============================
    # ===== Visual Perception =====
    # =============================
    def observe_environment(self, logger, current_step, images_list):        
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict
    
    # ===================================
    # ===== Progress Estimation =========
    # ===================================
    def save_history(self, logger, current_step, next_vp, thought, curr_observe, nav_history):
        # ===== get obervation summary =====
        direction_id = int(curr_observe.split("Direction Viewpoint")[0].replace("Direction","").strip())
        direction = DIRECTIONS[direction_id]
        curr_observe = "Scene Description"+curr_observe.split("Scene Description")[1]
        observation = f"Direction {direction} " + self.llm.gpt_infer(OBSERVATION_SUMMARY['system'], OBSERVATION_SUMMARY['user'].format(curr_observe))
        # ===== get thought summary =====
        thought = self.llm.gpt_infer(THOUGHT_SUMMARY['system'], THOUGHT_SUMMARY['user'].format(thought))
        thought = thought.replace("Thought: ", "")
        # ===== get nav history =====
        nav_history.append({
            "step": current_step,
            "viewpoint": next_vp,
            "observation": observation,
            "thought": thought
        })
        logger.info(f"The history at current step is {nav_history}")
        return nav_history
    
    def review_history(self, logger, nav_history):
        nav_history_str = " -> ".join(["Step "+str(idx+1)+" Observation: "+item["observation"]+" Thought: "+item["thought"] for idx, item in enumerate(nav_history)])
        logger.info("History: " + nav_history_str)
        return nav_history_str
    
    def estimate_completion(self, logger, actions, landmarks, history_traj):
        response = self.llm.gpt_infer(COMPLETION_ESTIMATION['system'], COMPLETION_ESTIMATION['user'].format(history_traj, landmarks, actions))
        if "Decision" in response:
            logger.info("Decision " + response)
            if "Decision:" in response:
                return response.split("Decision:")[1].strip()
            else:
                return response.split("Decision")[1].strip()
        else:
            return response
    
    # =================================
    # ===== Move to next position =====
    # =================================
    def move_to_next_vp(self, logger, instruction, landmarks, history_traj, observation, observe_dict, images=None):    
        break_flag = True
        for i in range(2): # retry twice
            effective_prediction, thought_list, completion_estimations = [], [], []
            # batch_responses = self.llm.gpt_infer(NAVIGATOR['system'], 
            #                                       NAVIGATOR['user'].format(observe_dict.keys(), current_step, instruction,
            #                                                                actions, landmarks, history_traj, estimation, observation),
            #                                       num_output=3)
            batch_responses = self.llm.gpt_infer_with_images(MAPGPT_NAVIGATOR['system'], 
                                                             MAPGPT_NAVIGATOR['user'].format(observe_dict.keys(), instruction, 
                                                                                             landmarks, history_traj, observation), 
                                                             images=images,
                                                             num_output=2)
            for decision_reasoning in batch_responses:
                if "Prediction:" not in decision_reasoning:
                    continue
                logger.info(f"================retry id {i} in pred_vp==========")
                logger.info(decision_reasoning)

                # Parse the unified response with completion estimation
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                remaining_text = decision_reasoning.split("Prediction:")[1].strip()

                # Extract prediction and completion estimation
                if "Completion Estimation:" in remaining_text:
                    pred_vp = remaining_text.split("Completion Estimation:")[0].strip().replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")
                    completion_est = remaining_text.split("Completion Estimation:")[1].strip()
                else:
                    pred_vp = remaining_text.replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")
                    completion_est = "Unknown"

                effective_prediction.append(pred_vp)
                thought_list.append(pred_thought)
                completion_estimations.append(completion_est)

        return effective_prediction, thought_list, completion_estimations, break_flag
    
    def move_to_next_vp_single(self, logger, instruction, landmarks, history_traj, observation, observe_dict, images=None, next_instruction=None):
        # Set default next instruction if not provided
        if next_instruction is None:
            next_instruction = 'Stop.'

        # Capture system and user prompts for debug visualization
        system_prompt = MAPGPT_NAVIGATOR['system']
        user_prompt = MAPGPT_NAVIGATOR['user'].format(observe_dict.keys(), instruction, landmarks, history_traj, next_instruction, observation)

        # Store GPT interaction data for visualization
        gpt_interaction = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': None,
            'metadata': {
                'model': getattr(self.llm, 'model', 'unknown'),
                'num_images': len(images) if images else 0
            }
        }

        decision_reasoning = self.llm.gpt_infer_with_images(
            system_prompt,
            user_prompt,
            images=images)

        # Update with response
        gpt_interaction['response'] = decision_reasoning

        decision_reasoning = decision_reasoning.replace("**", "")
        if "Prediction:" not in decision_reasoning:
            logger.error(f"No Prediction in decision reasoning")
            # Try again
            decision_reasoning = self.llm.gpt_infer_with_images(
                system_prompt,
                user_prompt,
                images=images)

            # Update with retry response
            gpt_interaction['response'] = decision_reasoning
            gpt_interaction['metadata']['retry'] = True

            decision_reasoning = decision_reasoning.replace("**", "")
            if "Prediction:" not in decision_reasoning:
                next_vp, observe_description = random.choice(list(observe_dict.items()))
                logger.warning(f"Random choice a next predicted action {next_vp}")
                gpt_interaction['metadata']['random_fallback'] = True
                return next_vp, observe_description, gpt_interaction

        logger.info(decision_reasoning)

        # Parse the unified response with completion estimation
        # Use the LAST "Prediction:" to handle models that output multiple predictions in thinking
        prediction_parts = decision_reasoning.split("Prediction:")
        if len(prediction_parts) > 1:
            # Everything before the last "Prediction:" is the thought
            pred_thought = "Prediction:".join(prediction_parts[:-1]).strip()
            remaining_text = prediction_parts[-1].strip()
        else:
            pred_thought = ""
            remaining_text = decision_reasoning.strip()

        # Extract prediction and completion estimation
        if "Completion Estimation:" in remaining_text:
            pred_vp = remaining_text.split("Completion Estimation:")[0].strip().replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")
            completion_est = remaining_text.split("Completion Estimation:")[1].strip()
        else:
            pred_vp = remaining_text.replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")
            completion_est = "Unknown"

        # Extract numeric viewpoint ID from prediction (handles verbose LLM responses)
        # Look for patterns like "Direction 9", "Viewpoint 9", or just a number
        numeric_match = re.search(r'(?:Direction|Viewpoint)?\s*(\d+)', pred_vp)
        if numeric_match:
            pred_vp = numeric_match.group(1)
        else:
            # If no number found, log warning and try to extract any digit
            logger.warning(f"Could not extract viewpoint ID from prediction: {pred_vp[:100]}...")
            digit_match = re.search(r'\d+', pred_vp)
            if digit_match:
                pred_vp = digit_match.group()

        return pred_vp, pred_thought, completion_est, gpt_interaction
    
    # =========================
    # ===== Test Decision =====
    # =========================
    def thought_fusion(self, logger, predictions, thoughts):
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            matched_dict[pred].append(thought)
            
        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            one_thought = self.llm.gpt_infer(THOUGHT_FUSION['system'], THOUGHT_FUSION['user'].format(multiple_thoughts))
            logger.info(f"Pred viewpoint ID: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought 
        return matched_dict 
    
    def test_decisions(self, logger, fused_pred_thought, observation, instruction, error_number, observe_dict):
        try:
            for fused_key in list(fused_pred_thought.keys()):
                if len(fused_key) > 2:
                    fused_pred_thought.pop(fused_key)
                    
            if not fused_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(fused_pred_thought.keys()) == 1:
                for key, value in fused_pred_thought.items():
                    return key, value, error_number
            else:
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in fused_pred_thought.items()])
                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp = self.llm.gpt_infer(DECISION_TEST['system'], DECISION_TEST['user'].format(fused_pred_thought.keys(), observation, instruction, fused_pred_thought_))
                    logger.info(f"Next predicted action is {next_vp}")
                    if re.search(r'\D', next_vp):
                        next_vp = re.search(r'\d+', next_vp).group() 
        
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {fused_pred_thought[next_vp]}")
            return next_vp, fused_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            
            if error_number >= 2: 
                error_number = 0 
                if fused_pred_thought and all(len(key) < 2 for key in fused_pred_thought):
                    logger.info(f"Random choice a next predicted action {next_vp} in fused_pred_thought, error number reset to {error_number}")
                    next_vp, _ = random.choice(list(fused_pred_thought.items()))
                    return next_vp, fused_pred_thought[next_vp], error_number
                else:
                    next_vp, observe_description = random.choice(list(observe_dict.items()))
                    logger.info(f"Random choice a next predicted action {next_vp}, error number reset to {error_number}")
                    return next_vp, observe_description, error_number
            return "error_next_vp", "None", error_number

    def judge(self, logger, chosen_images, instruction):
        """
        Judge if the navigation path (sequence of images) follows the instruction.
        Args:
            chosen_images: list of PIL.Image RGB images (in order of visitation)
            instruction: the navigation instruction
        Returns:
            Tuple of (judgement, confidence, reasoning, judge_interaction)
            Judgement can be 'Yes', 'Stay', 'Backtrack', or 'Look Around'.
            Confidence is a score from 0-10.
            judge_interaction contains the prompts and response for visualization.
        """
        # Capture system and user prompts for debug visualization
        system_prompt = JUDGE_PROMPT['system']
        user_prompt = JUDGE_PROMPT['user'].format(instruction)
        images_dict = {str(i): {'rgb': img} for i, img in enumerate(chosen_images)}

        # Store judge interaction data for visualization
        judge_interaction = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': None,
            'metadata': {
                'model': getattr(self.llm, 'model', 'unknown'),
                'num_images': len(chosen_images),
                'instruction': instruction
            }
        }

        response = self.llm.gpt_infer_with_images(system_prompt, user_prompt, images_dict)

        # Update with response
        judge_interaction['response'] = response

        logger.info(f"Instruction: {instruction} \nLength of chosen images: {len(chosen_images)} \n{response}")

        response = response.replace("**", "")

        # Extract confidence score
        confidence = 5  # Default confidence
        if "Confidence:" in response:
            try:
                confidence_str = response.split("Confidence:")[1].split("\n")[0].strip()
                confidence = int(confidence_str)
                confidence = max(0, min(10, confidence))  # Clamp between 0-10
            except:
                logger.error(f"Could not parse confidence score, using default: 5")

        if "Judgement:" not in response:
            logger.error(f"No Judgement in response")
            return "Yes", confidence, "None", judge_interaction

        # Extract reasoning and judgement
        parts = response.split("Judgement:")
        if len(parts) >= 2:
            reasoning_part = parts[0].strip()
            judgement_part = parts[1].strip().replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")

            # Extract reasoning (everything before Confidence)
            if "Confidence:" in reasoning_part:
                reasoning = reasoning_part.split("Confidence:")[0].strip()
            else:
                reasoning = reasoning_part
        else:
            reasoning = "None"
            judgement_part = "Yes"

        return judgement_part, confidence, reasoning, judge_interaction
    
    def look_around(self, logger, current_step, images_dict, instruction, landmarks, history_traj):
        """
        Look around at candidate viewpoints to gather more information.
        Args:
            logger: logger object
            current_step: current step number
            images_dict: dictionary of available viewpoints and their images
            instruction: current instruction
            landmarks: landmarks to look for
            history_traj: navigation history
        Returns:
            enhanced observation and observation dictionary
        """
        logger.info("========== Look Around Mode ==========")
        logger.info("Exploring candidate viewpoints to gather more information")
        
        # Get observations from all available viewpoints
        all_observations = []
        all_observe_dict = {}
        
        for viewpoint_id, viewpoint_data in images_dict.items():
            logger.info(f"Exploring viewpoint {viewpoint_id}")
            observation, observe_dict = self.observe_environment(logger, current_step, {viewpoint_id: viewpoint_data})
            all_observations.extend(observation)
            all_observe_dict[viewpoint_id] = observe_dict[viewpoint_id]
        
        # Create a comprehensive observation summary
        comprehensive_observation = " ".join(all_observations)
        logger.info(f"Comprehensive observation from all viewpoints: {comprehensive_observation}")
        
        return comprehensive_observation, all_observe_dict
