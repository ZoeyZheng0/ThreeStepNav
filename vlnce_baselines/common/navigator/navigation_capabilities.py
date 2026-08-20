"""
Navigation Capabilities Documentation
This module contains the implementation code for navigation abilities
that the decision agent can read and understand.
"""

from typing import Dict, List, Any, Optional
import inspect
import ast


class NavigationCapabilities:
    """
    Repository of navigation capability implementations.
    The agent can read these to understand what each action actually does.
    """
    
    @staticmethod
    def continue_to_next_instruction(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move to the next sub-instruction in the navigation plan.
        
        Implementation:
        1. Increment current_action_idx
        2. Reset navigation history for fresh start
        3. Clear visual memory for new sub-goal
        4. Update context state
        
        Returns:
            Updated context after moving to next instruction
        """
        context['current_action_idx'] += 1
        context['nav_history'] = []
        context['history_traj'] = "Step 0 start position."
        context['state'] = 'navigating'
        return context
    
    @staticmethod
    def stay_with_current_instruction(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Continue working on the current sub-instruction.
        
        Implementation:
        1. Maintain current_action_idx
        2. Preserve navigation history
        3. Continue accumulating visual evidence
        4. Update confidence metrics
        
        Returns:
            Context with updated confidence but same instruction
        """
        context['attempts_on_current'] = context.get('attempts_on_current', 0) + 1
        context['state'] = 'navigating'
        return context
    
    @staticmethod
    def backtrack_to_previous(context: Dict[str, Any], env_actions_history: List) -> Dict[str, Any]:
        """
        Return to the previous position by reversing last action.
        
        Implementation:
        1. Pop last action from history
        2. Calculate reverse action (opposite angle, same distance)
        3. Remove last visual observation
        4. Decrement step counter
        5. Execute reverse movement in environment
        
        Returns:
            Context after backtracking
        """
        import math
        
        if len(env_actions_history) > 0:
            last_action = env_actions_history.pop()
            
            # Create reverse action
            reverse_action = {
                'action': {
                    'action': 4,  # Move action
                    'action_args': {
                        'angle': (last_action['action']['action_args']['angle'] + math.pi) % (2 * math.pi) - math.pi,
                        'distance': last_action['action']['action_args']['distance']
                    }
                }
            }
            
            # Update context
            if len(context.get('nav_history', [])) > 0:
                context['nav_history'].pop()
            if len(context.get('chosen_images', [])) > 0:
                context['chosen_images'].pop()
            
            context['backtrack_action'] = reverse_action
            context['state'] = 'backtracking'
            
        return context
    
    @staticmethod
    def look_around_comprehensive(context: Dict[str, Any], 
                                 all_viewpoints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather comprehensive visual information from all viewpoints.
        
        Implementation:
        1. Iterate through all available viewpoints
        2. Perform detailed analysis of each view
        3. Identify landmarks in each direction
        4. Build spatial map of surroundings
        5. Update observation dictionary with enhanced descriptions
        6. Calculate best next viewpoint based on comprehensive data
        
        Returns:
            Context with enhanced observations from all viewpoints
        """
        enhanced_observations = {}
        landmark_locations = {}
        
        for vp_id, vp_data in all_viewpoints.items():
            # Analyze each viewpoint
            enhanced_observations[vp_id] = {
                'base_observation': vp_data,
                'enhanced': True,
                'analysis_type': 'comprehensive'
            }
            
            # Track landmark positions
            if 'landmarks' in vp_data:
                landmark_locations[vp_id] = vp_data['landmarks']
        
        context['enhanced_observations'] = enhanced_observations
        context['landmark_map'] = landmark_locations
        context['state'] = 'exploring'
        context['look_around_completed'] = True
        
        return context


class CapabilityInspector:
    """
    Allows the agent to inspect and understand capability implementations.
    """
    
    @staticmethod
    def get_capability_code(capability_name: str) -> str:
        """
        Get the source code of a capability for agent understanding.
        
        Args:
            capability_name: Name of the capability
            
        Returns:
            Source code as string
        """
        capabilities = NavigationCapabilities()
        
        capability_map = {
            'continue': capabilities.continue_to_next_instruction,
            'stay': capabilities.stay_with_current_instruction,
            'backtrack': capabilities.backtrack_to_previous,
            'look_around': capabilities.look_around_comprehensive
        }
        
        if capability_name.lower() in capability_map:
            func = capability_map[capability_name.lower()]
            return inspect.getsource(func)
        
        return f"Capability '{capability_name}' not found"
    
    @staticmethod
    def get_capability_description(capability_name: str) -> Dict[str, str]:
        """
        Get structured description of a capability.
        
        Args:
            capability_name: Name of the capability
            
        Returns:
            Dictionary with description, implementation details, and effects
        """
        capabilities = NavigationCapabilities()
        
        descriptions = {
            'continue': {
                'name': 'Continue to Next Instruction',
                'purpose': 'Progress to the next sub-instruction when current is completed',
                'implementation': inspect.getsource(capabilities.continue_to_next_instruction),
                'effects': [
                    'Increments action index',
                    'Resets navigation history',
                    'Clears visual memory',
                    'Starts fresh for new sub-goal'
                ],
                'when_to_use': 'When current landmarks have been found and instruction is satisfied',
                'side_effects': 'Loses context from current instruction'
            },
            'stay': {
                'name': 'Stay with Current Instruction',
                'purpose': 'Continue working on the current sub-instruction',
                'implementation': inspect.getsource(capabilities.stay_with_current_instruction),
                'effects': [
                    'Maintains current action index',
                    'Preserves navigation history',
                    'Continues accumulating evidence',
                    'Increments attempt counter'
                ],
                'when_to_use': 'When current instruction is not yet satisfied',
                'side_effects': 'May lead to loops if stuck'
            },
            'backtrack': {
                'name': 'Backtrack to Previous Position',
                'purpose': 'Undo last movement and return to previous position',
                'implementation': inspect.getsource(capabilities.backtrack_to_previous),
                'effects': [
                    'Reverses last physical movement',
                    'Removes last history entry',
                    'Removes last visual observation',
                    'Provides opportunity to try different path'
                ],
                'when_to_use': 'When current path seems wrong or dead-end reached',
                'side_effects': 'Loses progress, may increase total steps'
            },
            'look_around': {
                'name': 'Look Around Comprehensively',
                'purpose': 'Gather detailed information from all viewpoints',
                'implementation': inspect.getsource(capabilities.look_around_comprehensive),
                'effects': [
                    'Analyzes all available viewpoints',
                    'Builds spatial map',
                    'Identifies all visible landmarks',
                    'Provides comprehensive scene understanding'
                ],
                'when_to_use': 'When confused or need more information to decide',
                'side_effects': 'Takes additional time, no physical movement'
            }
        }
        
        return descriptions.get(capability_name.lower(), {
            'error': f"Capability '{capability_name}' not found"
        })
    
    @staticmethod
    def analyze_capability_logic(capability_name: str) -> Dict[str, Any]:
        """
        Analyze the logic flow of a capability using AST.
        
        Args:
            capability_name: Name of the capability
            
        Returns:
            Analysis of the capability's logic flow
        """
        code = CapabilityInspector.get_capability_code(capability_name)
        
        try:
            tree = ast.parse(code)
            
            # Extract key operations
            operations = []
            conditions = []
            returns = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Track assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            operations.append(f"Assigns to: {target.id}")
                        elif isinstance(target, ast.Subscript):
                            if isinstance(target.value, ast.Name):
                                operations.append(f"Updates: {target.value.id}[...]")
                
                elif isinstance(node, ast.If):
                    # Track conditions
                    conditions.append("Has conditional logic")
                
                elif isinstance(node, ast.Return):
                    # Track returns
                    returns.append("Returns modified context")
            
            return {
                'capability': capability_name,
                'operations': operations,
                'has_conditions': len(conditions) > 0,
                'returns': returns,
                'complexity': 'simple' if len(operations) < 5 else 'moderate' if len(operations) < 10 else 'complex'
            }
            
        except Exception as e:
            return {
                'capability': capability_name,
                'error': str(e)
            }


class CapabilityExecutor:
    """
    Executes navigation capabilities with full context.
    """
    
    def __init__(self):
        self.capabilities = NavigationCapabilities()
        self.inspector = CapabilityInspector()
    
    def execute(self, capability_name: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute a navigation capability.
        
        Args:
            capability_name: Name of capability to execute
            context: Current navigation context
            **kwargs: Additional arguments for the capability
            
        Returns:
            Updated context after execution
        """
        capability_map = {
            'continue': self.capabilities.continue_to_next_instruction,
            'stay': self.capabilities.stay_with_current_instruction,
            'backtrack': lambda ctx: self.capabilities.backtrack_to_previous(
                ctx, kwargs.get('env_actions_history', [])
            ),
            'look_around': lambda ctx: self.capabilities.look_around_comprehensive(
                ctx, kwargs.get('all_viewpoints', {})
            )
        }
        
        if capability_name.lower() in capability_map:
            return capability_map[capability_name.lower()](context)
        
        raise ValueError(f"Unknown capability: {capability_name}")
    
    def explain_execution(self, capability_name: str) -> str:
        """
        Explain what will happen when a capability is executed.
        
        Args:
            capability_name: Name of capability
            
        Returns:
            Explanation of execution effects
        """
        description = self.inspector.get_capability_description(capability_name)
        
        if 'error' in description:
            return description['error']
        
        explanation = f"""
Capability: {description['name']}
Purpose: {description['purpose']}

What happens when executed:
{chr(10).join(f"- {effect}" for effect in description['effects'])}

When to use: {description['when_to_use']}
Side effects: {description['side_effects']}
        """
        
        return explanation.strip()