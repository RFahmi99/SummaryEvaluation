"""
Utility functions for the evaluation pipeline
"""

import json
import csv
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_test_data_from_json(filepath: str) -> List[Dict[str, str]]:
    """
    Load test cases from JSON file

    Expected format:
    [
        {
            "source_text": "...",
            "summary": "...",
            "reference": "..." (optional)
        }
    ]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} test cases from {filepath}")
    return data


def load_test_data_from_csv(filepath: str) -> List[Dict[str, str]]:
    """
    Load test cases from CSV file

    Expected columns: source_text, summary, reference (optional)
    """
    test_cases = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'source_text': row['source_text'],
                'summary': row['summary'],
                'reference': row.get('reference', '')
            })

    logger.info(f"Loaded {len(test_cases)} test cases from {filepath}")
    return test_cases


def save_results_to_json(results: List[Any], filepath: str):
    """Save evaluation results to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert EvaluationResult objects to dicts if needed
    if hasattr(results[0], 'to_dict'):
        results = [r.to_dict() for r in results]

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filepath}")


def save_results_to_csv(results: List[Any], filepath: str):
    """Save evaluation results to CSV file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert to dicts if needed
    if hasattr(results[0], 'to_dict'):
        results = [r.to_dict() for r in results]

    # Extract field names from first result
    fieldnames = list(results[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {filepath}")


def generate_summary_report(results: List[Any]) -> Dict[str, Any]:
    """
    Generate aggregate statistics from evaluation results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    # Convert to dicts if needed
    if hasattr(results[0], 'to_dict'):
        results = [r.to_dict() for r in results]

    total = len(results)
    passed = sum(1 for r in results if r.get('passed_all_checks', False))

    # Aggregate scores
    metrics = [
        'similarity_score',
        'factual_consistency_score',
        'relevance_score',
        'coherence_score',
        'fluency_score',
        'fairness_score'
    ]

    aggregates = {}
    for metric in metrics:
        values = [r.get(metric) for r in results if r.get(metric) is not None]
        if values:
            aggregates[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }

    # Safety statistics
    safety_passed = sum(1 for r in results if r.get('safety_passed', True))
    pii_detected_count = sum(1 for r in results if r.get('pii_detected'))

    report = {
        'total_samples': total,
        'passed': passed,
        'failed': total - passed,
        'pass_rate': passed / total if total > 0 else 0,
        'safety_pass_rate': safety_passed / total if total > 0 else 0,
        'pii_detection_rate': pii_detected_count / total if total > 0 else 0,
        'metric_statistics': aggregates
    }

    return report


def print_summary_report(report: Dict[str, Any]):
    """Print formatted summary report"""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY REPORT")
    print("="*70)

    print(f"\nTotal Samples: {report['total_samples']}")
    print(f"Passed: {report['passed']} ({report['pass_rate']:.1%})")
    print(f"Failed: {report['failed']} ({(1-report['pass_rate']):.1%})")

    print(f"\nSafety Pass Rate: {report['safety_pass_rate']:.1%}")
    print(f"PII Detection Rate: {report['pii_detection_rate']:.1%}")

    print("\nMetric Statistics:")
    print("-"*70)

    for metric, stats in report['metric_statistics'].items():
        metric_name = metric.replace('_', ' ').title()
        print(f"\n{metric_name}:")
        print(f"  Mean:  {stats['mean']:.3f}")
        print(f"  Min:   {stats['min']:.3f}")
        print(f"  Max:   {stats['max']:.3f}")
        print(f"  Count: {stats['count']}")

    print("\n" + "="*70)


def compare_summaries(summary1: str, summary2: str, source: str, 
                     pipeline) -> Dict[str, Any]:
    """
    Compare two summaries of the same source text

    Returns:
        Dictionary with comparison results
    """
    result1 = pipeline.evaluate_summary(summary1, source)
    result2 = pipeline.evaluate_summary(summary2, source)

    comparison = {
        'summary1': {
            'text': summary1,
            'scores': result1.to_dict()
        },
        'summary2': {
            'text': summary2,
            'scores': result2.to_dict()
        },
        'winner': {}
    }

    # Determine winner for each metric
    metrics = [
        'similarity_score',
        'factual_consistency_score',
        'relevance_score',
        'coherence_score',
        'fluency_score',
        'fairness_score'
    ]

    for metric in metrics:
        score1 = result1.to_dict().get(metric)
        score2 = result2.to_dict().get(metric)

        if score1 is not None and score2 is not None:
            if score1 > score2:
                comparison['winner'][metric] = 'summary1'
            elif score2 > score1:
                comparison['winner'][metric] = 'summary2'
            else:
                comparison['winner'][metric] = 'tie'

    return comparison


def filter_results(results: List[Any], 
                  min_score: float = 0.7,
                  metric: str = 'factual_consistency_score') -> List[Any]:
    """
    Filter results by minimum score threshold

    Args:
        results: List of evaluation results
        min_score: Minimum acceptable score
        metric: Which metric to filter on

    Returns:
        Filtered list of results
    """
    if hasattr(results[0], 'to_dict'):
        results = [r.to_dict() for r in results]

    filtered = [
        r for r in results 
        if r.get(metric) is not None and r.get(metric) >= min_score
    ]

    logger.info(f"Filtered {len(results)} -> {len(filtered)} results "
                f"(threshold: {metric} >= {min_score})")

    return filtered


def export_for_labeling(results: List[Any], output_path: str,
                       include_predictions: bool = True):
    """
    Export results in format suitable for human labeling/review

    Creates a CSV with summaries and optionally model predictions
    for manual quality assessment
    """
    if hasattr(results[0], 'to_dict'):
        results = [r.to_dict() for r in results]

    labeled_data = []

    for i, result in enumerate(results):
        row = {
            'id': i,
            'source_text': result['source_text'][:500] + '...',  # Truncate
            'summary': result['summary'],
            'human_rating': '',  # To be filled by human
            'human_notes': ''
        }

        if include_predictions:
            row['predicted_relevance'] = result.get('relevance_score', '')
            row['predicted_coherence'] = result.get('coherence_score', '')
            row['predicted_factual'] = result.get('factual_consistency_score', '')

        labeled_data.append(row)

    save_results_to_csv(labeled_data, output_path)
    logger.info(f"Exported {len(labeled_data)} samples for labeling to {output_path}")


def calculate_agreement(human_ratings: List[float], 
                       model_scores: List[float]) -> Dict[str, float]:
    """
    Calculate agreement between human ratings and model scores

    Returns:
        Dictionary with correlation and other metrics
    """
    import numpy as np
    from scipy import stats

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(human_ratings, model_scores)

    # Spearman correlation (rank-based)
    spearman_r, spearman_p = stats.spearmanr(human_ratings, model_scores)

    # Mean absolute error
    mae = np.mean(np.abs(np.array(human_ratings) - np.array(model_scores)))

    return {
        'pearson_correlation': pearson_r,
        'pearson_pvalue': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_pvalue': spearman_p,
        'mean_absolute_error': mae,
        'n_samples': len(human_ratings)
    }


# Quick access functions
def quick_evaluate(summary: str, source: str) -> Dict[str, Any]:
    """
    Quick evaluation returning the complete evaluation dictionary.
    """
    from src.evaluation.evaluation import SummaryEvaluationPipeline

    pipeline = SummaryEvaluationPipeline(
        enable_safety=True,
        enable_telemetry=False
    )

    result = pipeline.evaluate_summary(summary, source)

    # Return the full dictionary so the CSV handler has all keys
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully")

    # Demo: Generate mock report
    mock_results = [
        {
            'passed_all_checks': True,
            'similarity_score': 0.85,
            'factual_consistency_score': 0.90,
            'relevance_score': 0.88,
            'coherence_score': 0.82,
            'fluency_score': 0.87,
            'fairness_score': 0.91,
            'safety_passed': True,
            'pii_detected': None
        },
        {
            'passed_all_checks': False,
            'similarity_score': 0.65,
            'factual_consistency_score': 0.60,
            'relevance_score': 0.70,
            'coherence_score': 0.68,
            'fluency_score': 0.75,
            'fairness_score': 0.85,
            'safety_passed': True,
            'pii_detected': ['PERSON', 'EMAIL']
        }
    ]

    report = generate_summary_report(mock_results)
    print_summary_report(report)
