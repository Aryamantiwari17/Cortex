import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import json
import time

class LLMEvaluator:
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'token_count': [],
            'answer_relevancy': [],
            'factual_accuracy': [],
            'format_adherence': [],
            'memory_usage': []
        }
        
    def measure_response_time(self, func):
        """Decorator to measure response time of model calls"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            self.metrics['response_time'].append(end_time - start_time)
            return result
        return wrapper

    def evaluate_response(self, 
                         question: str,
                         ground_truth: str,
                         model_response: str,
                         expected_format: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Evaluate a single model response across multiple dimensions
        """
        # Evaluate answer relevancy (simple keyword matching)
        relevancy_score = self._calculate_relevancy(question, model_response)
        self.metrics['answer_relevancy'].append(relevancy_score)
        
        # Evaluate factual accuracy against ground truth
        accuracy_score = self._calculate_accuracy(ground_truth, model_response)
        self.metrics['factual_accuracy'].append(accuracy_score)
        
        # Check format adherence if format is specified
        if expected_format:
            format_score = self._check_format(model_response, expected_format)
            self.metrics['format_adherence'].append(format_score)
        
        return {
            'relevancy': relevancy_score,
            'accuracy': accuracy_score,
            'format': format_score if expected_format else None
        }

    def _calculate_relevancy(self, question: str, response: str) -> float:
        """
        Calculate relevancy score based on keyword matching
        """
        question_keywords = set(question.lower().split())
        response_keywords = set(response.lower().split())
        overlap = len(question_keywords.intersection(response_keywords))
        return overlap / len(question_keywords)

    def _calculate_accuracy(self, ground_truth: str, response: str) -> float:
        """
        Calculate accuracy score comparing response to ground truth
        """
        # Implement more sophisticated comparison logic here
        # This is a simple example using word overlap
        truth_words = set(ground_truth.lower().split())
        response_words = set(response.lower().split())
        overlap = len(truth_words.intersection(response_words))
        return overlap / len(truth_words)

    def _check_format(self, response: str, expected_format: Dict[str, Any]) -> float:
        """
        Check if response adheres to expected format
        """
        try:
            response_json = json.loads(response)
            format_matches = all(key in response_json for key in expected_format.keys())
            return 1.0 if format_matches else 0.0
        except json.JSONDecodeError:
            return 0.0

    def run_benchmark(self, 
                     test_cases: List[Dict[str, Any]],
                     langchain_model,
                     llamaindex_model) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive benchmark comparing LangChain and LlamaIndex implementations
        """
        results = {
            'langchain': {'avg_response_time': 0, 'avg_relevancy': 0, 'avg_accuracy': 0},
            'llamaindex': {'avg_response_time': 0, 'avg_relevancy': 0, 'avg_accuracy': 0}
        }
        
        for test_case in tqdm(test_cases):
            # Test LangChain
            langchain_result = self.evaluate_model_response(
                langchain_model,
                test_case['question'],
                test_case['ground_truth'],
                test_case.get('expected_format')
            )
            
            # Test LlamaIndex
            llamaindex_result = self.evaluate_model_response(
                llamaindex_model,
                test_case['question'],
                test_case['ground_truth'],
                test_case.get('expected_format')
            )
            
            # Aggregate results
            for metric in ['response_time', 'relevancy', 'accuracy']:
                results['langchain'][f'avg_{metric}'] += langchain_result[metric]
                results['llamaindex'][f'avg_{metric}'] += llamaindex_result[metric]
        
        # Calculate averages
        n = len(test_cases)
        for model in results:
            for metric in results[model]:
                results[model][metric] /= n
                
        return results

    def generate_report(self, benchmark_results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate detailed comparison report
        """
        report = ["Model Comparison Report", "=" * 20]
        
        metrics = ['avg_response_time', 'avg_relevancy', 'avg_accuracy']
        for metric in metrics:
            report.append(f"\n{metric.replace('_', ' ').title()}:")
            report.append("-" * 40)
            for model in benchmark_results:
                report.append(f"{model}: {benchmark_results[model][metric]:.3f}")
        
        # Add winner summary
        report.append("\nOverall Performance Summary:")
        report.append("-" * 40)
        
        winner_count = {'langchain': 0, 'llamaindex': 0}
        for metric in metrics:
            if benchmark_results['langchain'][metric] > benchmark_results['llamaindex'][metric]:
                winner_count['langchain'] += 1
            else:
                winner_count['llamaindex'] += 1
        
        winner = max(winner_count.items(), key=lambda x: x[1])[0]
        report.append(f"Overall winner: {winner.upper()} " 
                     f"(won {winner_count[winner]}/{len(metrics)} metrics)")
        
        return "\n".join(report)

# Example usage
def example_usage():
    # Create test cases
    test_cases = [
    ]
    
    evaluator = LLMEvaluator()
    
    # Replace these with your actual model implementations
    langchain_model = None  # Your LangChain implementation
    llamaindex_model = None  # Your LlamaIndex implementation
    
    results = evaluator.run_benchmark(test_cases, langchain_model, llamaindex_model)
    report = evaluator.generate_report(results)
    print(report)