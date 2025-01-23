import sys
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.adapters.provider import ProviderAdapter
from src.adapters.google import GoogleAdapter

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePredictionTest:
    def __init__(self, model_info_path: str, llm_adapter: ProviderAdapter):
        """Initialize prediction test"""
        self.model_info_path = Path(model_info_path)
        self.llm = llm_adapter
        self.load_model_info()
        
    def load_model_info(self):
        """Load model information"""
        with open(self.model_info_path, 'r') as f:
            self.model_info = json.load(f)
            
    def create_prediction_prompt(self, sequence: List[float], include_context: bool = True) -> str:
        """Create prediction prompt"""
        prompt = []
        
        # Add sequence with clear formatting
        prompt.append("You are analyzing a time series sequence.")
        prompt.append("\nPrevious values:")
        for i, val in enumerate(sequence):
            prompt.append(f"{i}: {val:.4f}")
            
        if include_context:
            prompt.append("\nContext:")
            prompt.append("- This is sensor data from a smartphone")
            prompt.append("- Values represent physical movements")
            prompt.append("- Data is normalized to [-1, 1] range")
            
        prompt.append("\nBased on this pattern, predict the next value.")
        prompt.append("Format your response exactly as:")
        prompt.append("Prediction: [number]")
        prompt.append("Explanation: [your reasoning]")
        
        return "\n".join(prompt)
    
    def parse_prediction(self, response: str) -> Optional[float]:
        """Parse prediction value from response"""
        try:
            # Find prediction line
            for line in response.split('\n'):
                if line.lower().startswith('prediction:'):
                    # Extract number
                    value = line.split(':')[1].strip()
                    return float(value)
        except Exception as e:
            logger.error(f"Failed to parse prediction: {str(e)}")
        return None
        
    def run_test(self, test_sequences: List[List[float]], 
                 true_values: List[float],
                 sample_size: int = 10) -> Dict:
        """Run prediction test"""
        # Sample sequences if needed
        if len(test_sequences) > sample_size:
            indices = np.random.choice(len(test_sequences), sample_size, replace=False)
            test_sequences = [test_sequences[i] for i in indices]
            true_values = [true_values[i] for i in indices]
            
        results = {
            'predictions': [],
            'true_values': true_values,
            'errors': [],
            'responses': []
        }
        
        for i, sequence in enumerate(test_sequences):
            logger.info(f"Processing sequence {i+1}/{len(test_sequences)}")
            
            # Get prediction
            prompt = self.create_prediction_prompt(sequence)
            response = self.llm.chat_completion(prompt)
            prediction = self.parse_prediction(response)
            
            # Store results
            results['predictions'].append(prediction)
            results['responses'].append(response)
            if prediction is not None:
                error = abs(prediction - true_values[i])
                results['errors'].append(error)
            else:
                results['errors'].append(None)
                
        return results
    
def load_test_data(data_dir: str) -> tuple:
    """Load test sequences from HAR dataset"""
    data_path = Path(data_dir) / 'test' / 'X_test.txt'
    sequence_length = 10
    
    # Load data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            data.append(values[0])  # Take first feature
            
    # Create sequences
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
        
    return sequences, targets

def main():
    try:
        # Setup
        data_dir = Path("data/har/UCI HAR Dataset")
        model_dir = Path("results/2024-12-28/LTC_FullyConnected_64_lr0.001_batch10_noise0_length24")
        model_info = model_dir / 'nn_info.json'
        
        # Load data
        logger.info("Loading test data...")
        sequences, targets = load_test_data(data_dir)
        logger.info(f"Loaded {len(sequences)} test sequences")
        
        # Initialize test
        llm = GoogleAdapter()
        test = SimplePredictionTest(model_info, llm)
        
        # Run test
        logger.info("Running prediction test...")
        results = test.run_test(sequences, targets, sample_size=10)
        
        # Save results
        output_dir = Path("results/prediction_tests") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            # Convert numpy values to native Python types
            clean_results = {
                'predictions': [float(x) if x is not None else None for x in results['predictions']],
                'true_values': [float(x) for x in results['true_values']],
                'errors': [float(x) if x is not None else None for x in results['errors']],
                'responses': results['responses']
            }
            json.dump(clean_results, f, indent=2)
            
        # Print summary
        valid_errors = [e for e in results['errors'] if e is not None]
        if valid_errors:
            print(f"\nResults Summary:")
            print(f"Average Error: {np.mean(valid_errors):.4f}")
            print(f"Success Rate: {len(valid_errors)}/{len(results['predictions'])}")
        
    except Exception as e:
        logger.error(f"Error in prediction test: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()