"""
Decision Agent for Navigation using LangChain
Based on OpenAI's Agent Design Patterns
Analyzes navigation progress and makes decisions
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field
from typing import Optional
import re


class NavigationDecision(Enum):
    """Possible navigation decisions"""
    CONTINUE = "Continue"  # Move to next sub-instruction
    STAY = "Stay"  # Continue with current sub-instruction
    BACKTRACK = "Backtrack"  # Go back to previous position
    LOOK_AROUND = "Look Around"  # Explore more viewpoints


@dataclass
class DecisionContext:
    """Context for decision making"""
    chosen_images: List[Image.Image]
    current_action: str
    current_landmarks: List[str]
    actions_completed: str
    history_trajectory: str
    current_step: int
    total_actions: int
    current_action_idx: int
    image_descriptions: Optional[List[str]] = None  # Descriptions for each image


class DecisionOutput(BaseModel):
    """Structured output for navigation decisions"""
    decision: str = Field(description="Navigation decision: Continue/Stay/Backtrack/Look Around")
    confidence: float = Field(description="Confidence score from 0 to 10")
    reasoning: str = Field(description="Detailed reasoning for the decision")


class NavigationDecisionParser(BaseOutputParser):
    """Custom parser for navigation decisions"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the LLM output to extract decision components"""
        text = text.replace("**", "").strip()
        
        # Extract decision
        decision = "Stay"
        if "Decision:" in text:
            decision_line = text.split("Decision:")[1].split("\n")[0].strip()
            decision = decision_line.split(",")[0].strip()
        
        # Extract confidence
        confidence = 5.0
        if "Confidence:" in text:
            try:
                conf_line = text.split("Confidence:")[1].split("\n")[0].strip()
                conf_str = re.findall(r'\d+\.?\d*', conf_line)[0]
                confidence = float(conf_str)
                confidence = max(0, min(10, confidence))
            except:
                pass
        
        # Extract reasoning
        reasoning = ""
        if "Reasoning:" in text:
            reasoning = text.split("Reasoning:")[1].strip()
        elif "Thought:" in text:
            reasoning = text.split("Thought:")[1].strip()
        else:
            reasoning = text
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def get_format_instructions(self) -> str:
        return """Format your response as:
Decision: [Continue/Stay/Backtrack/Look Around]
Confidence: [0-10]
Reasoning: [Your detailed reasoning]"""


class LangChainLLMWrapper(LLM):
    """Wrapper to make existing LLM client compatible with LangChain"""
    
    llm_client: Any = None
    
    def __init__(self, llm_client, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = llm_client
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the wrapped LLM client."""
        # Extract system and user prompts if structured
        if "System:" in prompt and "User:" in prompt:
            system = prompt.split("User:")[0].replace("System:", "").strip()
            user = prompt.split("User:")[1].strip()
            return self.llm_client.gpt_infer(system, user)
        return self.llm_client.gpt_infer("You are a helpful assistant.", prompt)


class NavigationDecisionAgent:
    """
    Agent that decides navigation strategy based on visual history.
    Implements the single-agent pattern using LangChain.
    """

    def __init__(self, llm_client, enabled_abilities=None, logger: Optional[logging.Logger] = None):
        """
        Initialize the decision agent with LangChain components.

        Args:
            llm_client: LLM client for inference
            enabled_abilities: List of enabled meta abilities (continue, stay, backtrack, look_around)
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger("DecisionAgent")

        # Set enabled meta abilities
        default_abilities = ["continue", "stay", "backtrack", "look_around"]
        self.enabled_abilities = enabled_abilities if enabled_abilities is not None else default_abilities
        self.logger.info(f"Enabled meta abilities: {self.enabled_abilities}")
        
        # Wrap LLM for LangChain compatibility
        self.llm = LangChainLLMWrapper(llm_client)
        
        # Initialize parser
        self.parser = NavigationDecisionParser()
        
        # Create decision chain
        self.decision_chain = self._create_decision_chain()

        # Create analysis tools
        self.tools = self._create_tools()
        
        # Memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def _generate_examples_for_enabled_abilities(self) -> str:
        """Generate in-context learning examples only for enabled abilities"""
        examples = "## In-Context Learning Examples for Each Meta Ability:\n\n"

        if "continue" in self.enabled_abilities:
            examples += """### CONTINUE Examples (Move to next sub-instruction):
Case 1: "Walk through the doorway" completed, next is "Turn left at the hallway"
- Observation: Doorway successfully passed, now in a hallway with left/right options
- Decision: Continue - doorway task achieved, move to hallway navigation
- Confidence: 9/10

Case 2: "Go up the stairs" completed, next is "Enter the second door on your right"
- Observation: At top of stairs, corridor with multiple doors visible
- Decision: Continue - stairs climbed successfully, ready for door-finding task
- Confidence: 8/10

Case 3: "Exit the bedroom" completed, next is "Walk down the hallway to the kitchen"
- Observation: Standing outside bedroom door, long hallway visible ahead
- Decision: Continue - bedroom exited, proceed with hallway navigation
- Confidence: 9/10

"""

        if "stay" in self.enabled_abilities:
            examples += """### STAY Examples (Continue with current instruction):
Case 1: "Go to the living room with the fireplace"
- Observation: In a living room but no fireplace visible yet
- Decision: Stay - correct room type but missing key landmark (fireplace)
- Confidence: 6/10

Case 2: "Walk past three doors on your left"
- Observation: Passed two doors so far, third door visible ahead
- Decision: Stay - task partially complete (2/3 doors), continue counting
- Confidence: 7/10

Case 3: "Navigate to the end of the hallway"
- Observation: Midway through hallway, end not yet reached
- Decision: Stay - still in progress toward hallway end
- Confidence: 7/10

"""

        if "backtrack" in self.enabled_abilities:
            examples += """### BACKTRACK Examples (Return to previous position):
Case 1: "Turn right at the intersection"
- Observation: Turned left by mistake, wrong corridor
- Decision: Backtrack - wrong turn taken, need to return to intersection
- Confidence: 9/10

Case 2: "Enter the room with blue walls"
- Observation: Entered room with white walls, no blue visible
- Decision: Backtrack - wrong room entered, return to hallway
- Confidence: 8/10

Case 3: "Go through the glass door"
- Observation: Went through wooden door instead, different room than expected
- Decision: Backtrack - incorrect door chosen, need to find glass door
- Confidence: 9/10

"""

        if "look_around" in self.enabled_abilities:
            examples += """### LOOK AROUND Examples (Gather more information):
Case 1: "Find the room with the piano"
- Observation: At intersection with multiple room entrances, unclear which has piano
- Decision: Look Around - need to check multiple viewpoints for piano visibility
- Confidence: 5/10

Case 2: "Go toward the kitchen"
- Observation: Multiple paths available, kitchen direction unclear
- Decision: Look Around - explore viewpoints to identify kitchen indicators
- Confidence: 4/10

Case 3: "Locate the staircase going down"
- Observation: In large open area, staircase not immediately visible
- Decision: Look Around - scan environment comprehensively for stairs
- Confidence: 5/10

"""

        return examples

    def _create_decision_chain(self) -> LLMChain:
        """Create the main decision-making chain"""

        # Create list of available abilities for the prompt
        abilities_list = []
        for ability in self.enabled_abilities:
            abilities_list.append(ability.title())
        abilities_str = ", ".join(abilities_list)

        # Generate examples for enabled abilities
        examples_text = self._generate_examples_for_enabled_abilities()

        decision_template = f"""System: You are a navigation decision agent analyzing a sequence of images from a navigation path.

Your role is to:
1. Analyze whether the navigation is following the given instructions
2. Assess confidence in the current progress
3. Decide the next action from these available options: {abilities_str}

IMPORTANT: You can ONLY choose from these enabled abilities: {abilities_str}

You have access to tools that let you:
- Read the actual implementation code of each navigation capability
- Understand what each action does internally
- Analyze the logic and side effects of each decision

{examples_text}

Current Context:
- Instruction: {{instruction}}
- Actions completed: {{actions_completed}}
- Current landmarks: {{landmarks}}
- Navigation history: {{history}}
- Number of images: {{num_images}}

Available Capabilities (you can read their code using tools):
- Continue: Move to next sub-instruction (resets history)
- Stay: Keep working on current instruction (preserves context)
- Backtrack: Reverse last movement (undoes progress)
- Look Around: Gather comprehensive information (no movement)

Analyze the navigation sequence considering:
- Visual continuity and logical progression
- Landmark visibility and achievement
- Path correctness relative to instructions
- Need for more information or correction
- The actual implementation effects of each capability

Use the in-context examples above to guide your decision-making process.

{{format_instructions}}

User: Based on the visual sequence and understanding of capability implementations, what is your navigation decision?"""
        
        prompt = PromptTemplate(
            input_variables=["instruction", "actions_completed", "landmarks", 
                           "history", "num_images"],
            template=decision_template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.parser,
            verbose=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create analysis tools for the agent including code reading"""
        
        # Import capability inspector
        from vlnce_baselines.common.navigator.navigation_capabilities import (
            CapabilityInspector, CapabilityExecutor
        )
        
        self.capability_inspector = CapabilityInspector()
        self.capability_executor = CapabilityExecutor()
        
        def analyze_visual_continuity(images_info: str) -> str:
            """Analyze visual continuity between navigation images"""
            return f"Visual continuity analysis: The image sequence shows logical progression with consistent environmental features."
        
        def identify_landmarks(image_info: str, target_landmarks: str) -> str:
            """Identify which landmarks are visible"""
            return f"Landmark detection: Analyzing for landmarks: {target_landmarks}"
        
        def assess_progress(current_step: int, total_steps: int) -> str:
            """Assess navigation progress"""
            progress = (current_step / total_steps) * 100 if total_steps > 0 else 0
            return f"Progress assessment: {progress:.1f}% complete (step {current_step}/{total_steps})"
        
        def read_capability_code(capability_name: str) -> str:
            """Read the implementation code of a navigation capability"""
            code = self.capability_inspector.get_capability_code(capability_name)
            return f"Implementation code for {capability_name}:\n{code}"
        
        def understand_capability(capability_name: str) -> str:
            """Get detailed understanding of what a capability does"""
            description = self.capability_inspector.get_capability_description(capability_name)
            if 'error' in description:
                return description['error']
            
            return f"""
Capability: {description['name']}
Purpose: {description['purpose']}
Effects: {', '.join(description['effects'])}
When to use: {description['when_to_use']}
Side effects: {description['side_effects']}
"""
        
        def analyze_capability_logic(capability_name: str) -> str:
            """Analyze the logic flow of a capability"""
            analysis = self.capability_inspector.analyze_capability_logic(capability_name)
            return f"""
Logic analysis for {capability_name}:
- Operations: {', '.join(analysis.get('operations', []))}
- Has conditions: {analysis.get('has_conditions', False)}
- Complexity: {analysis.get('complexity', 'unknown')}
"""
        
        def explain_capability_execution(capability_name: str) -> str:
            """Explain what happens when a capability is executed"""
            return self.capability_executor.explain_execution(capability_name)
        
        return [
            Tool(
                name="AnalyzeVisualContinuity",
                func=analyze_visual_continuity,
                description="Analyze visual continuity in image sequence"
            ),
            Tool(
                name="IdentifyLandmarks", 
                func=identify_landmarks,
                description="Identify visible landmarks in images"
            ),
            Tool(
                name="AssessProgress",
                func=assess_progress,
                description="Assess overall navigation progress"
            ),
            Tool(
                name="ReadCapabilityCode",
                func=read_capability_code,
                description="Read the actual implementation code of a navigation capability (continue/stay/backtrack/look_around)"
            ),
            Tool(
                name="UnderstandCapability",
                func=understand_capability,
                description="Get detailed understanding of what a navigation capability does"
            ),
            Tool(
                name="AnalyzeCapabilityLogic",
                func=analyze_capability_logic,
                description="Analyze the logic flow and complexity of a capability"
            ),
            Tool(
                name="ExplainCapabilityExecution",
                func=explain_capability_execution,
                description="Explain what will happen when a capability is executed"
            )
        ]
    
    def make_decision(self, context: DecisionContext) -> Tuple[NavigationDecision, float, str]:
        """
        Make a navigation decision based on context using LangChain.
        
        Args:
            context: Decision context with images and navigation state
            
        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        self.logger.info("========== Navigation Decision Agent (LangChain) ==========")
        
        # Prepare context for chain
        chain_input = {
            "instruction": context.current_action,
            "actions_completed": context.actions_completed,
            "landmarks": ", ".join(context.current_landmarks) if context.current_landmarks else "None",
            "history": context.history_trajectory,
            "num_images": len(context.chosen_images)
        }
        
        # If we have images, we need to use the original client for multimodal
        if context.chosen_images:
            # For multimodal, fall back to direct LLM call
            images_dict = {
                str(i): {'rgb': img} 
                for i, img in enumerate(context.chosen_images)
            }
            
            # Create enhanced prompt with image descriptions
            user_prompt = self.decision_chain.prompt.format(**chain_input).split("User:")[1].strip()
            if context.image_descriptions:
                descriptions_text = "\n\nImage sequence descriptions:\n"
                for i, desc in enumerate(context.image_descriptions):
                    descriptions_text += f"Image {i}: {desc}\n"
                user_prompt = descriptions_text + "\n" + user_prompt
                self.logger.info(f"Added image descriptions to prompt: {descriptions_text}")
            
            response = self.llm_client.gpt_infer_with_images(
                self.decision_chain.prompt.format(**chain_input).split("User:")[0].replace("System:", "").strip(),
                user_prompt,
                images_dict
            )
            
            # Parse with our parser
            parsed = self.parser.parse(response)
        else:
            # Use chain for text-only decisions
            parsed = self.decision_chain.run(**chain_input)
        
        # Convert to enum
        decision = self._string_to_decision(parsed["decision"])
        confidence = parsed["confidence"]
        reasoning = parsed["reasoning"]
        
        # Apply decision rules
        decision = self._apply_decision_rules(decision, confidence, context)
        
        # Log decision
        self.logger.info(f"Decision: {decision.value}")
        self.logger.info(f"Confidence: {confidence}/10")
        self.logger.info(f"Reasoning: {reasoning}")
        
        # Store in memory for context
        self.memory.save_context(
            {"input": f"Context at step {context.current_step}"},
            {"output": f"Decision: {decision.value} (confidence: {confidence})"}
        )
        
        return decision, confidence, reasoning

    def make_decision_with_capture(self, context: DecisionContext) -> Tuple[NavigationDecision, float, str, Dict]:
        """
        Make a navigation decision and capture GPT interaction data.

        Args:
            context: Decision context with images and navigation state

        Returns:
            Tuple of (decision, confidence, reasoning, interaction_data)
        """
        decision, confidence, reasoning = self.make_decision(context)

        # Convert images to base64 for storage
        import base64
        import io
        images_base64 = {}
        for i, img in enumerate(context.chosen_images):
            with io.BytesIO() as buf:
                img.save(buf, format='JPEG')
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                image_base64_url = f"data:image/jpeg;base64,{image_base64}"
                images_base64[str(i)] = image_base64_url

        # Capture the interaction that was used
        interaction_data = {
            'system_prompt': self.decision_chain.prompt.format(**{
                "instruction": context.current_action,
                "actions_completed": context.actions_completed,
                "landmarks": ", ".join(context.current_landmarks) if context.current_landmarks else "None",
                "history": context.history_trajectory,
                "num_images": len(context.chosen_images)
            }).split("User:")[0].replace("System:", "").strip(),
            'user_prompt': self.decision_chain.prompt.format(**{
                "instruction": context.current_action,
                "actions_completed": context.actions_completed,
                "landmarks": ", ".join(context.current_landmarks) if context.current_landmarks else "None",
                "history": context.history_trajectory,
                "num_images": len(context.chosen_images)
            }).split("User:")[1].strip(),
            'response': f"Decision: {decision.value}\nConfidence: {confidence}\nReasoning: {reasoning}",
            'images_base64': images_base64,
            'metadata': {
                'model': getattr(self.llm_client, 'model', 'unknown'),
                'method': 'make_decision',
                'num_images': len(context.chosen_images),
                'current_step': context.current_step
            }
        }

        return decision, confidence, reasoning, interaction_data

    def make_informed_decision(self, context: DecisionContext) -> Tuple[NavigationDecision, float, str]:
        """
        Make an informed decision by first understanding capability implementations.
        
        This method actively reads and analyzes the code of navigation capabilities
        before making a decision.
        
        Args:
            context: Decision context with images and navigation state
            
        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        self.logger.info("========== Informed Navigation Decision (with Code Analysis) ==========")
        
        # First, analyze the available capabilities
        capability_analysis = {}
        for capability in ['continue', 'stay', 'backtrack', 'look_around']:
            # Read and understand each capability
            code = self.tools[3].func(capability)  # ReadCapabilityCode
            understanding = self.tools[4].func(capability)  # UnderstandCapability
            logic = self.tools[5].func(capability)  # AnalyzeCapabilityLogic
            
            capability_analysis[capability] = {
                'code': code,
                'understanding': understanding,
                'logic': logic
            }
            
            self.logger.info(f"Analyzed capability '{capability}'")
        
        # Create enhanced prompt with capability understanding
        enhanced_prompt = f"""
I have analyzed the implementation of each navigation capability:

CONTINUE Implementation:
{capability_analysis['continue']['understanding']}
{capability_analysis['continue']['logic']}

STAY Implementation:
{capability_analysis['stay']['understanding']}
{capability_analysis['stay']['logic']}

BACKTRACK Implementation:
{capability_analysis['backtrack']['understanding']}
{capability_analysis['backtrack']['logic']}

LOOK_AROUND Implementation:
{capability_analysis['look_around']['understanding']}
{capability_analysis['look_around']['logic']}

## In-Context Learning Examples:

### CONTINUE (Resets history, moves to next sub-instruction):
- "Walk through doorway" done → "Turn left": Doorway passed, ready for turn = Continue (9/10)
- "Go upstairs" done → "Find door #2": At top, doors visible = Continue (8/10)
- "Exit room" done → "Go to kitchen": Outside room, hallway ahead = Continue (9/10)

### STAY (Preserves context, continues current):
- "Find fireplace room": In living room, no fireplace yet = Stay (6/10)
- "Pass 3 doors": Passed 2/3 doors = Stay (7/10)
- "Reach hallway end": Midway through = Stay (7/10)

### BACKTRACK (Reverses last action):
- "Turn right": Turned left instead = Backtrack (9/10)
- "Blue wall room": Entered white room = Backtrack (8/10)
- "Glass door": Went through wood door = Backtrack (9/10)

### LOOK AROUND (Explores viewpoints, no movement):
- "Find piano room": Multiple rooms, unclear = Look Around (5/10)
- "Go to kitchen": Multiple paths available = Look Around (4/10)
- "Find stairs down": Large area, not visible = Look Around (5/10)

Given this understanding of what each action actually does in the code,
and considering the current context:
- Instruction: {context.current_action}
- Landmarks to find: {', '.join(context.current_landmarks) if context.current_landmarks else 'None'}
- History: {context.history_trajectory}
- Images analyzed: {len(context.chosen_images)}

What is the most appropriate navigation decision?
Consider the actual code effects, not just the conceptual purpose.
Match your situation to the examples above.
"""
        
        # Make decision with enhanced understanding
        if context.chosen_images:
            images_dict = {
                str(i): {'rgb': img} 
                for i, img in enumerate(context.chosen_images)
            }
            
            # Add image descriptions if available
            final_prompt = enhanced_prompt
            if context.image_descriptions:
                descriptions_text = "\n\nImage sequence descriptions:\n"
                for i, desc in enumerate(context.image_descriptions):
                    descriptions_text += f"Image {i}: {desc}\n"
                final_prompt = descriptions_text + "\n" + enhanced_prompt
                self.logger.info(f"Added image descriptions to informed decision prompt")
            
            response = self.llm_client.gpt_infer_with_images(
                "You are a code-aware navigation decision agent that understands implementation details.",
                final_prompt + "\n\n" + self.parser.get_format_instructions(),
                images_dict
            )
        else:
            response = self.llm_client.gpt_infer(
                "You are a code-aware navigation decision agent that understands implementation details.",
                enhanced_prompt + "\n\n" + self.parser.get_format_instructions()
            )
        
        # Parse response
        parsed = self.parser.parse(response)
        decision = self._string_to_decision(parsed["decision"])
        confidence = parsed["confidence"]
        reasoning = f"[Code-Informed] {parsed['reasoning']}"
        
        # Apply decision rules
        decision = self._apply_decision_rules(decision, confidence, context)
        
        # Log informed decision
        self.logger.info(f"Informed Decision: {decision.value}")
        self.logger.info(f"Confidence: {confidence}/10")
        self.logger.info(f"Reasoning: {reasoning}")
        
        # Store in memory
        self.memory.save_context(
            {"input": f"Informed context at step {context.current_step} with code analysis"},
            {"output": f"Decision: {decision.value} (confidence: {confidence}) - Code-informed"}
        )
        
        return decision, confidence, reasoning

    def make_informed_decision_with_capture(self, context: DecisionContext) -> Tuple[NavigationDecision, float, str, Dict]:
        """
        Make an informed decision and capture GPT interaction data.

        Args:
            context: Decision context with images and navigation state

        Returns:
            Tuple of (decision, confidence, reasoning, interaction_data)
        """
        self.logger.info("========== Informed Navigation Decision (with Code Analysis) ==========")

        # First, analyze the available capabilities
        capability_analysis = {}
        for capability in ['continue', 'stay', 'backtrack', 'look_around']:
            # Read and understand each capability
            code = self.tools[3].func(capability)  # ReadCapabilityCode
            understanding = self.tools[4].func(capability)  # UnderstandCapability
            logic = self.tools[5].func(capability)  # AnalyzeCapabilityLogic

            capability_analysis[capability] = {
                'code': code,
                'understanding': understanding,
                'logic': logic
            }

            self.logger.info(f"Analyzed capability '{capability}'")

        # Create enhanced prompt with capability understanding
        enhanced_prompt = f"""
I have analyzed the implementation of each navigation capability:

CONTINUE Implementation:
{capability_analysis['continue']['understanding']}
{capability_analysis['continue']['logic']}

STAY Implementation:
{capability_analysis['stay']['understanding']}
{capability_analysis['stay']['logic']}

BACKTRACK Implementation:
{capability_analysis['backtrack']['understanding']}
{capability_analysis['backtrack']['logic']}

LOOK_AROUND Implementation:
{capability_analysis['look_around']['understanding']}
{capability_analysis['look_around']['logic']}

## In-Context Learning Examples:

### CONTINUE (Resets history, moves to next sub-instruction):
- "Walk through doorway" done → "Turn left": Doorway passed, ready for turn = Continue (9/10)
- "Go upstairs" done → "Find door #2": At top, doors visible = Continue (8/10)
- "Exit room" done → "Go to kitchen": Outside room, hallway ahead = Continue (9/10)

### STAY (Preserves context, continues current):
- "Find fireplace room": In living room, no fireplace yet = Stay (6/10)
- "Pass 3 doors": Passed 2/3 doors = Stay (7/10)
- "Reach hallway end": Midway through = Stay (7/10)

### BACKTRACK (Reverses last action):
- "Turn right": Turned left instead = Backtrack (9/10)
- "Blue wall room": Entered white room = Backtrack (8/10)
- "Glass door": Went through wood door = Backtrack (9/10)

### LOOK AROUND (Explores viewpoints, no movement):
- "Find piano room": Multiple rooms, unclear = Look Around (5/10)
- "Go to kitchen": Multiple paths available = Look Around (4/10)
- "Find stairs down": Large area, not visible = Look Around (5/10)

Given this understanding of what each action actually does in the code,
and considering the current context:
- Instruction: {context.current_action}
- Landmarks to find: {', '.join(context.current_landmarks) if context.current_landmarks else 'None'}
- History: {context.history_trajectory}
- Images analyzed: {len(context.chosen_images)}

What is the most appropriate navigation decision?
Consider the actual code effects, not just the conceptual purpose.
Match your situation to the examples above.
"""

        # Store system prompt for capture
        system_prompt = "You are a code-aware navigation decision agent that understands implementation details."

        # Make decision with enhanced understanding
        if context.chosen_images:
            images_dict = {
                str(i): {'rgb': img}
                for i, img in enumerate(context.chosen_images)
            }

            # Add image descriptions if available
            final_prompt = enhanced_prompt
            if context.image_descriptions:
                descriptions_text = "\n\nImage sequence descriptions:\n"
                for i, desc in enumerate(context.image_descriptions):
                    descriptions_text += f"Image {i}: {desc}\n"
                final_prompt = descriptions_text + "\n" + enhanced_prompt
                self.logger.info(f"Added image descriptions to informed decision prompt")

            response = self.llm_client.gpt_infer_with_images(
                system_prompt,
                final_prompt + "\n\n" + self.parser.get_format_instructions(),
                images_dict
            )
        else:
            response = self.llm_client.gpt_infer(
                system_prompt,
                enhanced_prompt + "\n\n" + self.parser.get_format_instructions()
            )

        # Parse response
        parsed = self.parser.parse(response)
        decision = self._string_to_decision(parsed["decision"])
        confidence = parsed["confidence"]
        reasoning = f"[Code-Informed] {parsed['reasoning']}"

        # Apply decision rules
        decision = self._apply_decision_rules(decision, confidence, context)

        # Log informed decision
        self.logger.info(f"Informed Decision: {decision.value}")
        self.logger.info(f"Confidence: {confidence}/10")
        self.logger.info(f"Reasoning: {reasoning}")

        # Store in memory
        self.memory.save_context(
            {"input": f"Informed context at step {context.current_step} with code analysis"},
            {"output": f"Decision: {decision.value} (confidence: {confidence}) - Code-informed"}
        )

        # Convert images to base64 for storage
        import base64
        import io
        images_base64 = {}
        for i, img in enumerate(context.chosen_images):
            with io.BytesIO() as buf:
                img.save(buf, format='JPEG')
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                image_base64_url = f"data:image/jpeg;base64,{image_base64}"
                images_base64[str(i)] = image_base64_url

        # Capture the interaction data
        interaction_data = {
            'system_prompt': system_prompt,
            'user_prompt': final_prompt + "\n\n" + self.parser.get_format_instructions() if context.chosen_images else enhanced_prompt + "\n\n" + self.parser.get_format_instructions(),
            'response': response,
            'images_base64': images_base64,
            'metadata': {
                'model': getattr(self.llm_client, 'model', 'unknown'),
                'method': 'make_informed_decision',
                'num_images': len(context.chosen_images),
                'current_step': context.current_step,
                'code_analysis_used': True
            }
        }

        return decision, confidence, reasoning, interaction_data

    def _string_to_decision(self, decision_str: str) -> NavigationDecision:
        """Convert string to NavigationDecision enum"""
        decision_str = decision_str.strip().upper()
        if "CONTINUE" in decision_str or "YES" in decision_str:
            return NavigationDecision.CONTINUE
        elif "BACKTRACK" in decision_str:
            return NavigationDecision.BACKTRACK
        elif "LOOK" in decision_str or "AROUND" in decision_str:
            return NavigationDecision.LOOK_AROUND
        else:
            return NavigationDecision.STAY
    
    def _parse_response(self, response: str) -> Tuple[NavigationDecision, float, str]:
        """
        Parse LLM response to extract decision components.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed decision, confidence, and reasoning
        """
        response = response.replace("**", "").strip()
        
        # Extract decision
        decision = NavigationDecision.STAY  # Default
        if "Decision:" in response:
            decision_str = response.split("Decision:")[1].split("\n")[0].strip()
            if "Continue" in decision_str or "Yes" in decision_str:
                decision = NavigationDecision.CONTINUE
            elif "Backtrack" in decision_str:
                decision = NavigationDecision.BACKTRACK
            elif "Look" in decision_str or "Around" in decision_str:
                decision = NavigationDecision.LOOK_AROUND
            else:
                decision = NavigationDecision.STAY
        
        # Extract confidence
        confidence = 5.0  # Default moderate confidence
        if "Confidence:" in response:
            try:
                conf_line = response.split("Confidence:")[1].split("\n")[0].strip()
                # Handle various formats: "7", "7/10", "7.5"
                conf_str = conf_line.split("/")[0].strip()
                confidence = float(conf_str)
                confidence = max(0, min(10, confidence))  # Clamp to [0, 10]
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse confidence from: {conf_line}")
        
        # Extract reasoning
        reasoning = ""
        if "Reasoning:" in response:
            reasoning_parts = response.split("Reasoning:")[1].strip()
            # Take everything after "Reasoning:" as the reasoning
            reasoning = reasoning_parts.split("Decision:")[0].strip() if "Decision:" in reasoning_parts else reasoning_parts
        elif "Thought:" in response:
            reasoning = response.split("Thought:")[1].strip()
        else:
            # Use entire response as reasoning if no specific section
            reasoning = response
        
        return decision, confidence, reasoning
    
    def _apply_decision_rules(self,
                             decision: NavigationDecision,
                             confidence: float,
                             context: DecisionContext) -> NavigationDecision:
        """
        Apply rule-based adjustments to the decision.

        Args:
            decision: Initial decision
            confidence: Confidence score
            context: Decision context

        Returns:
            Adjusted decision
        """
        # First, check if the decision is allowed by enabled abilities
        decision_str = decision.value.lower().replace(" ", "_")
        if decision_str not in self.enabled_abilities:
            self.logger.info(f"Decision '{decision.value}' not in enabled abilities {self.enabled_abilities}")
            # Fallback to first enabled ability
            fallback_decision = self._string_to_decision(self.enabled_abilities[0])
            self.logger.info(f"Falling back to {fallback_decision.value}")
            decision = fallback_decision

        # Low confidence -> Look around for more information (if enabled)
        if confidence < 5 and "look_around" in self.enabled_abilities:
            self.logger.info(f"Low confidence ({confidence}), overriding to LOOK_AROUND")
            return NavigationDecision.LOOK_AROUND
        elif confidence < 5 and "stay" in self.enabled_abilities:
            self.logger.info(f"Low confidence ({confidence}), LOOK_AROUND not available, using STAY")
            return NavigationDecision.STAY

        # If we're at the last action and decision is continue, change to stay (if enabled)
        if (decision == NavigationDecision.CONTINUE and
            context.current_action_idx == context.total_actions - 1 and
            "stay" in self.enabled_abilities):
            self.logger.info("At last action, changing CONTINUE to STAY")
            return NavigationDecision.STAY

        # If no images yet, stay to gather more information (if enabled)
        if len(context.chosen_images) < 2 and "stay" in self.enabled_abilities:
            self.logger.info("Insufficient visual history, staying")
            return NavigationDecision.STAY

        return decision
    
    def should_stop_navigation(self, 
                              decision: NavigationDecision,
                              context: DecisionContext) -> bool:
        """
        Determine if navigation should stop.
        
        Args:
            decision: Current decision
            context: Decision context
            
        Returns:
            True if navigation should stop
        """
        # Stop if we've completed all actions and decision is continue/stay
        if (context.current_action_idx == context.total_actions - 1 and
            decision in [NavigationDecision.CONTINUE, NavigationDecision.STAY]):
            return True
        
        # Stop if we've exceeded maximum steps
        if context.current_step >= 20:  # Or use config value
            return True
        
        return False


class EnhancedDecisionAgent(NavigationDecisionAgent):
    """
    Enhanced version with additional analysis capabilities.
    """
    
    def __init__(self, llm_client, spatial_client=None, logger=None):
        super().__init__(llm_client, logger)
        self.spatial = spatial_client
        
    def analyze_visual_continuity(self, images: List[Image.Image]) -> float:
        """
        Analyze visual continuity between images.
        
        Args:
            images: Sequence of navigation images
            
        Returns:
            Continuity score (0-1)
        """
        if len(images) < 2:
            return 1.0
        
        # Analyze progression through images
        prompt = """Analyze the visual continuity and logical progression of this image sequence.
        Rate from 0-1 how well these images form a coherent navigation path."""
        
        images_dict = {str(i): {'rgb': img} for i, img in enumerate(images)}
        
        response = self.llm.gpt_infer_with_images(
            "You are a visual continuity analyzer.",
            prompt,
            images_dict
        )
        
        try:
            # Extract score from response
            import re
            scores = re.findall(r'0\.\d+|1\.0', response)
            if scores:
                return float(scores[0])
        except:
            pass
        
        return 0.7  # Default moderate continuity
    
    def identify_landmarks_in_view(self, 
                                  image: Image.Image,
                                  target_landmarks: List[str]) -> List[str]:
        """
        Identify which target landmarks are visible in image.
        
        Args:
            image: Current view image
            target_landmarks: List of landmarks to find
            
        Returns:
            List of identified landmarks
        """
        if not target_landmarks:
            return []
        
        prompt = f"""Identify which of these landmarks are visible in the image:
        {', '.join(target_landmarks)}
        
        List only the landmarks that are clearly visible."""
        
        response = self.llm.gpt_infer_with_images(
            "You are a landmark detection assistant.",
            prompt,
            {"0": {"rgb": image}}
        )
        
        # Parse identified landmarks
        identified = []
        for landmark in target_landmarks:
            if landmark.lower() in response.lower():
                identified.append(landmark)
        
        return identified