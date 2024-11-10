#%%
!pip install openai

#%%

import openai

# openai.api_key = ''

# Function to query LLM
def query_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in neural networks and machine learning."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message['content']

# Example prompt
prompt = """
I am comparing two types of neural networks: classical (RNN, LSTM) and bio-inspired (Liquid Neural Network). 
Given the following model architectures:
- Liquid Neural Network with wiring: Fully Connected, 64 neurons.
- RNN with LSTM mid layer, 64 neurons.

I want to predict their performance on a noisy sinusoidal data generation task. Which architecture is likely to perform better and why?
"""
response = query_llm(prompt)
print("Pre-training analysis by LLM:\n", response)


# Post training prompting
explanation_comparison_prompt = f"""
Before training, you predicted that the Liquid Neural Network with Fully Connected wiring and 64 neurons would outperform 
an RNN with LSTM mid layer and 64 neurons on a noisy sinusoidal task. 
However, after training, the RNN performed better in terms of validation loss.
Can you explain why your prediction might have been incorrect based on the actual training results?
"""
explanation_response = query_llm(explanation_comparison_prompt)
print("Explanation comparison by LLM:\n", explanation_response)

# %%
