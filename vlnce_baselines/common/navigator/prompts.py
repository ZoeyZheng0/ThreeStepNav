# # Actions Decompsition
# ACTION_DETECTION = {
#     'system': "You are an action decomposition expert. Your task is to detect all actions in the given navigation instruction. You need to ensure the integrity of each action. Your answer must consist ONLY of a series of labled action phrases without begin sentence.",
#     'user': "Can you decompose actions in the instruction \"{}\"? Actions: "
# }

# Actions Decompsition
ACTION_DETECTION = {
    'system': "You are an action decomposition expert. Your task is to decompose the whole instruction into a series of sub-instructions and all actions in the given navigation instruction. You need to ensure the integrity of each action. You need to make sure the sub-instructions are complete, and include the details of the current environment if it is mentioned in the instruction. \
                Your answer must consist ONLY of a series of labled action phrases without begin sentence. \
                For each sub-instruction, it should involve at least one action, and all the description of the environment related to the same location. \
                A typical answer should involve 3 to 8 sub-instructions.",
    'user': "Can you decompose actions in the instruction \"{}\"? Actions: "
}

# Landmarks Extraction
LANDMARK_DETECTION = {
    'system': "You are a landmark extraction expert. Your task is to detect all landmarks in the given navigation instruction. You need to ensure the integrity of each landmarks. Your answer must consist ONLY of a series of labeled landmark phrases without other sentences.",
    'user': "Can you extract landmarks in the instruction \"{}\"? Landmarks: "
}

# Directions in Observation
DIRECTIONS = ["Front, range(left 15 to right 15)", "Font Left, range(left 15 to left 45)", "Left, range(left 45 to left 75)", "Left, range(left 75 to left 105)", "Rear Left, range(left 105 to left 135)", "Rear Left, range(left 135 to left 165)",
                    "Back, range(left 165 to right 165)", "Rear Right, range(right 135 to right 165)", "Right, range(right 105 to right 135)", "Right, range(right 75 to right 105)", "Front Right, range(right 45 to right 75)", "Front Right, range(right 15 to right 45)"]

# Summarize Observation
OBSERVATION_SUMMARY = {
    'system': "You are a trajectory summary expert. Your task is to simplify environment description as short and clear as possible. \
                                            You ONLY need to summarize in a single paragraph.",
    'user': "Given Environment Description \"{}\", Summarization:"
}

# Summarize Thought
THOUGHT_SUMMARY = {
    'system': "You are a trajectory summary expert. Your task is to simplify navigation thought process as short and clear as possible. \
                                            You ONLY need to summarize the what actions you did and what landmarks you passed in \"Thought\" using a single paragraph. Do NOT include Direction information. ",
    'user': "Given Thought Process \"{}\", Summarization:"
}

# # Estimate Completion
# COMPLETION_ESTIMATION = {
#     'system': "You are a completion estimation expert. Your task is to estimate the instruction have been executed or not based on navigation history and the aiming landmarks in the current instruction. \
#                 Your answer includes two parts: \"Thought\" and \"Decision\". You need to use \"Thought\" and \"Decision\" without any other symbols. \
#                 In the \"Thought\", you must follow procedures to analyze as detailed as possible what actions have been executed: \
#                 (1) What given landmarks of actions have appeared in the navigation history? \
#                 (2) Analyze the direction change at each step in the navigation history. \
#                 (3) Estimate the current instruction based on each step in the navigation history to check their completion. \
#                 In the \"Decision\", you must only write down 'Yes' or 'No' without other words. \
#                 You must strictly refer original actions in the given instruction to estimate.",
#     'user': "Given Navigation History \"{}\" and Landmarks \"[{}]\" in the instruction \"{}\", estimate the instruction have been executed or not."
# }

# # Main Navigator
# NAVIGATOR = {
#     'system': "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
#             I will give you one instruction and tell you landmarks. I will also give you navigation history and estimation of executed actions for reference. \
#             You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you. \
#             Each direction contains direction viewpoint ids you can move to. Your task is to predict moving to which direction viewpoint. \
#             In each prediction, direction 0 always represents your current orientation. Direction 1 represents the direction that is 30 degrees to the left of direction 0, Direction 2 represents the direction that is 60 degrees to the left of direction 0, Direction 3 represents the direction that is 90 degrees to the left of direction 0, Direction 4 represents the direction that is 120 degrees to the left of direction 0, Direction 5 represents the direction that is 150 degrees to the left of direction 0, Direction 6 represents the direction that is 180 degrees to the left of direction 0, Direction 7 represents the direction that is 150 degrees to the right of direction 0, Direction 8 represents the direction that is 120 degrees to the right of direction 0, Direction 9 represents the direction that is 90 degrees to the right of direction viewpoint ID 0, Direction 10 represents the direction that is 60 degrees to the right of direction 0, Direction 11 represents the direction that is 30 degrees to the right of direction 0 \
#             Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you. \
#             If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position. \
#             You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations. \
#             If you feel struggling to find the landmark or execute the action, you can try to execute the subsequent action and find the subsequent landmark. \
#             Your answer includes two parts: \"Thought\" and \"Prediction\". In the \"Thought\", you should think as detailed as possible following procedures: \
#             (1) The viewpoint ID you predicted must be one of the Direction Viewpoint ID in Candidate Viewpoint IDs List. The Candidate Viewpoint IDs List show the Direction Viewpoint ID that you should go. This means that there should be only a number after \"Prediction\" without any other words or characters . \
#             (2) Check whether the latest executed action has been completed by comparing current environment and landmark in the latest executed action. \
#             (3) Determine the action you should execute and landmark you should reach now. If the latest executed action have not been completed, \
#             you should continue to execute it. Otherwise, you should execute the next action in the given instruction. \
#             (4) Analyze which direction in the current environment is most suitable to execute the action you decide and explain your reason. \
#             (5) Predict moving to which direction viewpoint based on your thought process. \
#             (6) The \"Thought\" you predicted should be a single paragraph. \
#             (7) If you believe you have completed the instruction, you must still strictly follow the requirements to predict the next viewpoint in the \"Prediction\". \
#             (8) If you want to make a left turn, you usually need to select a viewpoint ID between 1 and 5. If you want to make a right turn, you usually need to select a viewpoint ID between 7 and 11. However, the viewpoint ID you predict must be within the Current Environment.\
#             (9) Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
#             Then, please make decision on the next viewpoint in the \"Prediction\". \
#             Your decision is very important, must make it very carefully. \
#             You need to double check the output in \"Prediction:\". The output must be in the Candidate Viewpoint IDs without any other words. \
#             You also need to double check the output in \"Thought\". The output must be a single paragraph",
#     'user': "Candidate Viewpoint IDs List: [{}] Step {} Instruction: {} ({}) Landmarks: {} Navigation History: {} \
#             Estimation of Executed Actions: {} Current Environment: {} -> Thought: ... Prediction: ... \
#             Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
#             Your output after \"Thought\" must be a single paragraph about why you choose this viewpoint id. "
# }

# MapGPT Navigator
MAPGPT_NAVIGATOR = {
    'system': "You are an embodied robot that navigates in the real world. \
            You need to explore between some places marked with IDs and ultimately find the destination to stop. \
            I will give you one instruction and tell you landmarks. I will also give you navigation history for reference. \
            You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you. \
            Each direction contains direction viewpoint ids you can move to. Your task is to predict moving to which direction viewpoint. \
            Each direction viewpoint has an image that you can see. \
            In each prediction, direction 0 always represents your current orientation. Direction 1 represents the direction that is 30 degrees to the left of direction 0, Direction 2 represents the direction that is 60 degrees to the left of direction 0, Direction 3 represents the direction that is 90 degrees to the left of direction 0, Direction 4 represents the direction that is 120 degrees to the left of direction 0, Direction 5 represents the direction that is 150 degrees to the left of direction 0, Direction 6 represents the direction that is 180 degrees to the left of direction 0, Direction 7 represents the direction that is 150 degrees to the right of direction 0, Direction 8 represents the direction that is 120 degrees to the right of direction 0, Direction 9 represents the direction that is 90 degrees to the right of direction viewpoint ID 0, Direction 10 represents the direction that is 60 degrees to the right of direction 0, Direction 11 represents the direction that is 30 degrees to the right of direction 0 \
            Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you. \
            If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position. \
            You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations. \
            For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress. \
            If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination. \
            Your answer includes four parts: \"Thought\", \"Distance\", \"Prediction\" and \"Completion Estimation\". In the \"Thought\", you should think as detailed as possible following procedures: \
            (1) The viewpoint ID you predicted must be one of the Direction Viewpoint ID in Candidate Viewpoint IDs List. The Candidate Viewpoint IDs List show the Direction Viewpoint ID that you should go. This means that there should be only a number after \"Prediction\" without any other words or characters . \
            (2) Analyze which direction in the current environment is most suitable to execute the instruction and explain your reason. \
            (3) You need to combine 'Instruction', 'Landmarks', your past 'Navigation History', 'Current Environment', and the provided images to think about what to do next, and complete your thinking into 'Thought'. \
            (4) Predict moving to which direction viewpoint based on your thought process. \
            (5) The \"Thought\" you predicted should be a single paragraph. \
            (6) If you believe you have completed the instruction, you must still strictly follow the requirements to predict the next viewpoint in the \"Prediction\". \
            (7) If you want to make a left turn, you usually need to select a viewpoint ID between 1 and 5. If you want to make a right turn, you usually need to select a viewpoint ID between 7 and 11. However, the viewpoint ID you predict must be within the Current Environment.\
            (8) Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
            Then, please make decision on the next viewpoint in the \"Prediction\". \
            Your decision is very important, must make it very carefully. \
            You need to double check the output in \"Prediction:\". The output must be in the Candidate Viewpoint IDs without any other words. \
            You also need to double check the output in \"Thought\". The output must be a single paragraph. \
            After finished all the above steps, you need to estimate the completion of the instruction based on the 'Instruction', 'Next instruction', 'Landmarks', your past 'Navigation History', 'Current Environment', and the provided images. \
            Please think carefully about the 'Distance' when you estimate the completion of the instruction. If your current distance to the destination is very far, you should answer 'No'. \
            If your current distance to the destination is close and you think you are ready to walk towards the landmarks of next instruction, you should answer 'Yes'.",
    'user': "Candidate Viewpoint IDs List: [{}] Instruction: {} Landmarks: {} Navigation History: {} Next instruction: {} \
            Current Environment: {} -> Thought: ... Distance: ... Prediction: ... Completion Estimation: ... "
}

# # Thought Fusion
# THOUGHT_FUSION = {
#     'system': "You are a thought fusion expert. Your task is to fuse given thought processes \
#                     into one thought. You need to reserve key information related to actions, landmarks, direction changes. You should only answer fused thought without other words.",
#     'user': "Can you help me fuse the thoughts leading to the same movement direction? The thoughts are :{}, Fused thought: "
# }

# # Test Decision
# DECISION_TEST = {
#     'system': "You are a decision testing expert. Your task is to evaluate the feasibility of each movement \
#                         prediction based on thought process and environment. Then, you will make a final decision about direction viewpoint ID without other words. \
#                             The answer should only be a number and within the candidate list.",
#     'user': "The candidate list: {}. Can you help me make a final decision? The Observation: {}, Navigation Instruction: {}, {}, Final Decision: "
# }

# Navigation Judge
JUDGE_PROMPT = {
    'system': (
        "You are a navigation judge. Determine if the provided navigation path (sequence of images) correctly follows the instruction. "
        "Sometimes, you may only receive one image, which means the agent has only taken one step so far. In this case, you should still make a decision about whether the path so far follows the instruction. "
        "Please note that the agent will not see the initial point of the navigation path. The first image is the first step of the agent has taken. "
        "Respond with 'Reasoning', 'Confidence', and 'Judgement'. "
        "The 'Reasoning' is the thinking process. Use this field to explain the thinking process behind the judgement. "
        "The 'Confidence' must be a score from 0 to 10, where 10 means 100% sure and 0 means not sure at all. "
        "The 'Judgement' must be one of 'Yes', 'Stay', 'Backtrack', or 'Look Around'. "
        "'Yes' means the navigation path correctly follows the instruction and it's time to move to the next sub-instruction. "
        "'Stay' means the navigation path has not finished the current sub-instruction, you need to stay in the current sub-instruction, and continue to follow by the current sub-instruction. "
        "'Backtrack' means the agent has gone in the wrong direction and needs to go back to the previous step to try a different path. "
        "'Look Around' means the agent should explore candidate viewpoints to gather more information before making a decision. "
        "If your confidence score is below 5, you should choose 'Look Around'. If your confidence score is 5 or above, you can choose 'Yes', 'Stay', or 'Backtrack'."
    ),
    'user': (
        "Instruction: {} Below are the images (in order) representing the navigation path taken by the agent."
    )
}
