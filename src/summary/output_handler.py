"""
Output handler module for saving results to CSV
"""
import csv
import os
from datetime import datetime
from typing import List, Dict

class OutputHandler:
    """Handles saving results to CSV format"""

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize output handler

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_to_csv(self, results: List[Dict], model_name: str, 
                    max_tokens: int, additional_info: Dict = None) -> str:
        """
        Save results to CSV file

        Args:
            results: List of result dictionaries
            model_name: Name of the LLM model used
            max_tokens: Max tokens parameter used
            additional_info: Additional metadata to include

        Returns:
            Path to saved CSV file
        """
        # Clean model name for filename
        clean_model_name = model_name.replace(':', '_').replace('/', '_')
        filename = f"{clean_model_name}_{max_tokens}.csv"
        filepath = os.path.join(self.output_dir, filename)

        # Prepare CSV headers
        headers = [
            'post_id', 'summarized_post', 'source_content', 'reference_summary',
            'time_taken_seconds', 'prompt_tokens', 'completion_tokens', 'total_tokens',
            'total_attempts', 'initial_passed_checks', 'final_passed_checks', 'improvement_context_used',
            'similarity_score', 'factual_consistency_score', 'relevance_score', 
            'coherence_score', 'fluency_score', 'fairness_score',
            'safety_passed', 'toxicity_score', 'passed_all_checks', 'failure_reasons'
        ]

        print(f"\nSaving results to: {filepath}")

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            for i, result in enumerate(results, 1):
                tox_score = result.get('toxicity_score')
                fail_reasons = result.get('failure_reasons', [])

                row = {
                    # Summary Data
                    'post_id': i,
                    'summarized_post': result.get('summary', ''),
                    'source_content': result.get('source', ''),
                    'reference_summary': result.get('reference', ''),
                    'time_taken_seconds': f"{result.get('time_taken', 0):.4f}",
                    'prompt_tokens': result.get('prompt_tokens', 0),
                    'completion_tokens': result.get('completion_tokens', 0),
                    'total_tokens': result.get('total_tokens', 0),
                    'total_attempts': result.get('total_attempts', 1),
                    'initial_passed_checks': result.get('initial_passed_checks', 'N/A'),
                    'final_passed_checks': result.get('passed_all_checks', 'N/A'),
                    'improvement_context_used': result.get('improvement_context_used', False),
                    
                    # Evaluation Metrics
                    'similarity_score': f"{result.get('similarity_score', 0):.4f}" if result.get('similarity_score') is not None else '',
                    'factual_consistency_score': f"{result.get('factual_consistency_score', 0):.4f}" if result.get('factual_consistency_score') is not None else '',
                    'relevance_score': f"{result.get('relevance_score', 0):.4f}" if result.get('relevance_score') is not None else '',
                    'coherence_score': f"{result.get('coherence_score', 0):.4f}" if result.get('coherence_score') is not None else '',
                    'fluency_score': f"{result.get('fluency_score', 0):.4f}" if result.get('fluency_score') is not None else '',
                    'fairness_score': f"{result.get('fairness_score', 0):.4f}" if result.get('fairness_score') is not None else '',
                    
                    # Security and Overall Status
                    'safety_passed': result.get('safety_passed', 'N/A'),
                    'toxicity_score': f"{tox_score:.4f}" if isinstance(tox_score, (int, float)) else 'N/A',
                    'passed_all_checks': result.get('passed_all_checks', 'N/A'),
                    'failure_reasons': "; ".join(fail_reasons) if fail_reasons else 'None'
                }
                writer.writerow(row)

        # Also save metadata file
        self._save_metadata(filepath, model_name, max_tokens, additional_info, results)

        print(f"✓ CSV saved successfully: {filepath}")
        print(f"  Total records: {len(results)}")

        return filepath

    def _save_metadata(self, csv_filepath: str, model_name: str, 
                      max_tokens: int, additional_info: Dict, results: List[Dict]):
        """Save metadata file alongside CSV"""
        metadata_path = csv_filepath.replace('.csv', '_metadata.txt')

        total_time = sum(r.get('time_taken', 0) for r in results)
        avg_time = total_time / len(results) if results else 0

        # Calculate aggregate token metrics
        total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in results)
        total_comp_tokens = sum(r.get('completion_tokens', 0) for r in results)
        total_tokens = sum(r.get('total_tokens', 0) for r in results)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("SUMMARIZATION PIPELINE METADATA\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV File: {os.path.basename(csv_filepath)}\n\n")

            f.write("Model Configuration:\n")
            f.write(f"  Model: {model_name}\n")
            f.write(f"  Max Tokens: {max_tokens}\n")

            if additional_info:
                f.write(f"  Temperature: {additional_info.get('temperature', 'N/A')}\n")
                f.write(f"  Random Seed: {additional_info.get('random_seed', 'N/A')}\n")
                f.write(f"  Dataset: {additional_info.get('dataset', 'N/A')}\n")
                f.write(f"  Number of Posts: {additional_info.get('num_posts', 'N/A')}\n")

            f.write("\nPerformance Statistics:\n")
            f.write(f"  Total Posts: {len(results)}\n")
            f.write(f"  Total Time: {total_time:.2f} seconds\n")

            f.write(f"  Total Prompt Tokens: {total_prompt_tokens}\n")
            f.write(f"  Total Completion Tokens: {total_comp_tokens}\n")
            f.write(f"  Total Tokens: {total_tokens}\n")

            f.write(f"  Average Time per Post: {avg_time:.2f} seconds\n")
            f.write(f"  Min Time: {min(r.get('time_taken', 0) for r in results):.2f} seconds\n")
            f.write(f"  Max Time: {max(r.get('time_taken', 0) for r in results):.2f} seconds\n")

        print(f"✓ Metadata saved: {metadata_path}")
