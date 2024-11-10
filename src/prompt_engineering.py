def generate_function_prediction_prompt(neuron_description, stimulus_info):
    """
    Generate a Chain-of-Thought prompt asking the LLM to reason about the neural network's behavior 
    given a stimulus.
    """
    prompt = f"""
    Consider the following neural network:
    {neuron_description}
    
    Given that stimulus is applied to neurons {stimulus_info['neurons']}, 
    predict which motor neurons will be activated and describe the signal flow.
    """
    return prompt

def generate_causal_reasoning_prompt(neuron_description, damage_info):
    """
    Generate a prompt asking the LLM to reason about the effect of network damage on behavior.
    """
    prompt = f"""
    A neural connection between neuron {damage_info['neuron1']} and {damage_info['neuron2']} is damaged.
    Predict the behavioral outcomes and compensatory mechanisms within the network.
    {neuron_description}
    """
    return prompt

def generate_adaptation_prompt(neuron_description, plasticity_data):
    """
    Generate a prompt asking the LLM to reason about network adaptation over time with plasticity.
    """
    prompt = f"""
    In this neural network, synaptic weights change over time due to learning:
    {plasticity_data}
    
    How will this influence the network's learning behavior? Predict the network's response to stimulus over time.
    {neuron_description}
    """
    return prompt
