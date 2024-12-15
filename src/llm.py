'''
...

    "graph": "Directed graph with nodes representing neurons and edges representing synaptic weights.",
    "matrix": "Weight matrix representation of neuron connections.",
    "equation": "Differential equations governing neuron state transitions.",
    "code": "Pseudocode representing the algorithmic flow of learning.",
    "diagram": "Visual diagram of layered network structure.",
....

'''
import json
import numpy as np

# Function to load the prompt template
def load_prompt_template(template_path):
    with open(template_path, "r") as file:
        return json.load(file)

# Generate a weight matrix as an example
def generate_weight_matrix(rows=4, cols=4):
    return np.random.rand(rows, cols)

# Representations of the architecture of learning
def get_representation(representation_type):
    alternatives = {
        "graph": read_graph(),
        "matrix": load_matrix(),
        "equation": define(),
        # "code": ,
        "diagram": read_image(),
    }
    return alternatives.get(representation_type, None)


def fill_prompt_template(template, weight_matrix, architecture_representation):
    # Fill in the template with variables
    filled_template = template["prompt"].format(
                        weight_matrix=str(weight_matrix),
                        architecture_of_learning=architecture_representation)
    return filled_template

def read_graph():
    print('TBD')
    return None

def load_matrix():
    print('TBD')
    return None

def define():
    print('TBD')
    return None

def read_image():
    print('TBD')
    return None


# demo usage
def main():
    # Load template
    template = load_prompt_template("prompt.json")
    
    # architecture representation
    representation_type = "graph"  # Change this to experiment with different types
    architecture_representation = get_representation(representation_type)
    
    # Fill in the template
    filled_prompt = fill_prompt_template(template, architecture_representation)
    
    print("Filled Prompt:\n")
    print(filled_prompt)

if __name__ == "__main__":
    main()
