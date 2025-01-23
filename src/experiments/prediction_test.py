import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
import logging
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.adapters.provider import ProviderAdapter

from src.adapters.google import GoogleAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.adapters.openai import OpenAIAdapter
from src.adapters.deepseek import DeepSeekAdapter
from src.adapters.groq import GroqAdapter
from src.utils.prediction_viz import create_report

# import keys from .env 
from dotenv import load_dotenv
load_dotenv(project_root / '.env')



# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionExperiment:
    def __init__(self, model_dir: str, llm_adapter):
        """
        Initialize prediction experiment.
        
        Args:
            model_dir: Directory containing trained model and its info
            llm_adapter: LLM adapter instance for predictions
        """
        self.model_dir = Path(model_dir)
        self.llm = llm_adapter
        self.load_model_info()

    def load_model_info(self):
        """Load model information from nn_info.json"""
        with open(self.model_dir / 'nn_info.json', 'r') as f:
            self.model_info = json.load(f)

    def format_sequence(self, sequence: List[float]) -> str:
        """Format sequence for better readability"""
        return '\n'.join([f"{i}: {val:.4f}" for i, val in enumerate(sequence)])

    def create_prediction_prompt(self, 
                               sequence: List[float], 
                               include_arch: bool = True,
                               include_model_pred: bool = True,
                               include_context: bool = True) -> str:
        """
        Create prompt for prediction task.
        """
        prompt = "You are analyzing time series data from a Human Activity Recognition dataset.\n\n"
        
        # Add sequence
        prompt += "Previous sequence values:\n"
        prompt += self.format_sequence(sequence) + "\n\n"
        
        # Add architecture info if requested
        if include_arch:
            prompt += "Neural Network Architecture:\n"
            prompt += f"Type: {self.model_info['architecture']}\n"
            prompt += f"Parameters: {self.model_info['parameters']}\n\n"
        
        # Add model prediction if requested
        if include_model_pred:
            prompt += "Neural Network's prediction:\n"
            prompt += f"{self.model_info['prediction']}\n\n"
        
        # Add context if requested
        if include_context:
            prompt += "Context: This data represents sensor readings from a smartphone during physical activities.\n"
            prompt += "The values are accelerometer and gyroscope measurements normalized to [-1, 1].\n\n"
        
        prompt += "Task: Based on this sequence, predict the next value that would maintain the pattern.\n"
        prompt += "Format your response as:\nPrediction: [number]\nExplanation: [your reasoning]\n"
        
        return prompt

    def parse_llm_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM response to extract prediction and explanation"""
        try:
            lines = response.strip().split('\n')
            prediction_line = [l for l in lines if l.startswith('Prediction:')][0]
            prediction = float(prediction_line.split(':')[1].strip())
            
            explanation_line = [l for l in lines if l.startswith('Explanation:')][0]
            explanation = explanation_line.split(':')[1].strip()
            
            return prediction, explanation
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {response}")
            logger.error(f"Error: {str(e)}")
            return None, str(e)

    def evaluate_predictions(self, 
                           true_values: List[float], 
                           predictions: List[float]) -> Dict[str, float]:
        """Calculate prediction metrics"""
        # Remove None values
        valid_pairs = [(t, p) for t, p in zip(true_values, predictions) if p is not None]
        if not valid_pairs:
            return {'mse': None, 'mae': None, 'correlation': None}
        
        true_array = np.array([t for t, _ in valid_pairs])
        pred_array = np.array([p for _, p in valid_pairs])
        
        return {
            'mse': float(np.mean((true_array - pred_array) ** 2)),
            'mae': float(np.mean(np.abs(true_array - pred_array))),
            'correlation': float(np.corrcoef(true_array, pred_array)[0, 1]) if len(true_array) > 1 else None
        }

    def run_experiment(self, 
                      test_sequences: List[List[float]], 
                      true_values: List[float],
                      variations: List[Dict[str, bool]] = None) -> Dict:
        """
        Run prediction experiment with different prompt variations.
        """
        if variations is None:
            variations = [
                {'include_arch': False, 'include_model_pred': False, 'include_context': False},
                {'include_arch': True, 'include_model_pred': False, 'include_context': False},
                {'include_arch': True, 'include_model_pred': True, 'include_context': False},
                {'include_arch': True, 'include_model_pred': True, 'include_context': True}
            ]
        
        results = {}
        for i, var in enumerate(variations):
            logger.info(f"Running variation {i+1}/{len(variations)}")
            predictions = []
            explanations = []
            
            for seq in test_sequences:
                prompt = self.create_prediction_prompt(seq, **var)
                response = self.llm.chat_completion(prompt)
                prediction, explanation = self.parse_llm_response(response)
                
                predictions.append(prediction)
                explanations.append(explanation)
            
            metrics = self.evaluate_predictions(true_values, predictions)
            
            results[f"variation_{i+1}"] = {
                'settings': var,
                'metrics': metrics,
                'predictions': predictions,
                'explanations': explanations
            }
        
        return results

def prepare_har_data(data_dir: str, sequence_length: int = 10) -> Tuple[List[List[float]], List[float]]:
    """Prepare HAR dataset for prediction experiments"""
    data_dir = Path(data_dir)
    
    # Load test data
    test_data = []
    with open(data_dir / 'test' / 'X_test.txt', 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            test_data.append(values)
    test_data = np.array(test_data)
    
    # Convert to sequences
    sequences = []
    targets = []
    
    for i in range(len(test_data) - sequence_length):
        seq = test_data[i:i+sequence_length, 0].tolist()  # Use first feature
        target = test_data[i+sequence_length, 0]
        sequences.append(seq)
        targets.append(target)
    
    return sequences[:100], targets[:100]  # Start with 100 sequences

def main():
    logger.info("Starting prediction experiment")
    
    # Setup paths
    base_dir = Path("C:/Users/corpa/Master/Thesis/results/2024-12-28")
    data_dir = Path("C:/Users/corpa/Master/Thesis/reasoning-with-LLMs/src/data/har/UCI HAR Dataset")
    
    # Use one model directory for testing
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    test_model_dir = model_dirs[0]
    logger.info(f"Using model from: {test_model_dir}")
    
    # Prepare data
    logger.info("Preparing HAR dataset")
    test_sequences, true_values = prepare_har_data(data_dir)
    logger.info(f"Prepared {len(test_sequences)} test sequences")

    # Initialize all LLM adapters
    adapters = {
        # 'gemini': GoogleAdapter(),
        'claude': AnthropicAdapter(),
        'gpt4': OpenAIAdapter(),
        'deepseek': DeepSeekAdapter(),
        # 'groq': GroqAdapter()
    }
    
    all_results = {}
    
    # Run experiments with each adapter
    for name, adapter in adapters.items():
        logger.info(f"\nRunning experiments with {name}")
        experiment = PredictionExperiment(test_model_dir, adapter)
        results = experiment.run_experiment(test_sequences, true_values)
        all_results[name] = results
    
    
    # Save results
    output_dir = Path("results/prediction_experiments") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "prediction_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Create visualizations
    create_report(str(results_file), str(output_dir))
    
    # Print summary
    print("\nExperiment Results Summary:")
    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        for var_name, var_results in model_results.items():
            print(f"  {var_name}:")
            print(f"    Settings: {var_results['settings']}")
            print(f"    Metrics: {var_results['metrics']}")

if __name__ == "__main__":
    main()