"""
DeepEval Integration Module
Provides integration with DeepEval framework as recommended in Phase 1
"""

from typing import List, Dict, Any, Optional
import json
import logging
import torch
import config

try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        GEval,
        BaseMetric
    )
    from deepeval.models.base_model import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logging.warning("DeepEval not installed. Install with: pip install deepeval")


class InstructJudge(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM wrapper using Ollama for Instruction-Tuned models.
    """

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def load_model(self):
        """DeepEval requirement, just returning the model name for Ollama"""
        return self.model_name

    def _clean_response(self, raw_content: str, prompt: str) -> str:
        """Helper to strip markdown and enforce expected DeepEval JSON keys."""
        raw_content = raw_content.strip()
        
        # 1. Strip markdown JSON wrappers if the LLM includes them
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:]
            
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
            
        raw_content = raw_content.strip()
        
        # 2. Try to parse and fix casing/schema issues dynamically
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, dict):
                # Lowercase all keys to fix {"Claims": ...} -> {"claims": ...}
                parsed = {k.lower(): v for k, v in parsed.items()}
                
                prompt_lower = prompt.lower()
                
                if "claims" in prompt_lower and "claims" not in parsed:
                    lists = [v for v in parsed.values() if isinstance(v, list)]
                    parsed["claims"] = lists[0] if lists else []
                    
                if "truths" in prompt_lower and "truths" not in parsed:
                    lists = [v for v in parsed.values() if isinstance(v, list)]
                    parsed["truths"] = lists[0] if lists else []
                    
                return json.dumps(parsed)
        except json.JSONDecodeError:
            pass # Return the cleaned string and let DeepEval's internal regex handle it
            
        return raw_content

    def generate(self, prompt: str) -> str:
        """Generate response using Ollama, enforcing JSON output"""
        import ollama
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict, analytical AI judge. Always format your output as valid JSON exactly as requested by the user prompt."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1},
            format="json" 
        )

        return self._clean_response(response['message']['content'], prompt)

    async def a_generate(self, prompt: str) -> str:
        """Async generate (using Ollama's async client)"""
        import ollama
        
        client = ollama.AsyncClient(timeout=1000.0)
        response = await client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict, analytical AI judge. Always format your output as valid JSON exactly as requested by the user prompt."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1},
            format="json"
        )
        
        return self._clean_response(response['message']['content'], prompt)

    def get_model_name(self) -> str:
        """Return model name"""
        return self.model_name


class DeepEvalPipeline:
    """
    Phase 1 orchestration using DeepEval framework
    Integrates with Prometheus 2 as the Judge LLM
    """

    def __init__(self, use_deepeval: bool = True):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval is required. Install with: pip install deepeval")

        self.use_deepeval = use_deepeval

        # Initialize judge model (updated to use the Ollama model name)
        if use_deepeval:
            self.judge_model = InstructJudge(model_name=config.ModelConfig.judge_model)
        else:
            self.judge_model = None

    def create_test_case(self,
                        input_text: str,
                        actual_output: str,
                        expected_output: Optional[str] = None,
                        context: Optional[List[str]] = None,
                        retrieval_context: Optional[List[str]] = None) -> LLMTestCase:
        """
        Create a DeepEval test case

        Args:
            input_text: The input/source text
            actual_output: Generated summary
            expected_output: Reference summary (optional)
            context: Additional context
            retrieval_context: Retrieved documents for RAG evaluation
        """
        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context
        )

    def create_metrics(self) -> List[BaseMetric]:
        """Create evaluation metrics using DeepEval"""
        metrics = []

        # Relevancy metric
        relevancy_metric = AnswerRelevancyMetric(
            threshold=config.ThresholdConfig.min_relevance,
            model=self.judge_model,
            include_reason=True,
            async_mode=False
        )
        metrics.append(relevancy_metric)

        # Faithfulness metric (factual consistency)
        faithfulness_metric = FaithfulnessMetric(
            threshold=config.ThresholdConfig.min_factual_consistency,
            model=self.judge_model,
            include_reason=True,
            async_mode=False
        )
        metrics.append(faithfulness_metric)

        # Hallucination detection
        # hallucination_metric = HallucinationMetric(
        #     threshold=config.ThresholdConfig.min_factual_consistency,
        #     model=self.judge_model,
        #     include_reason=True,
        #     async_mode=False
        # )
        # metrics.append(hallucination_metric)

        # Custom G-Eval for coherence
        coherence_metric = GEval(
            name="Coherence",
            criteria="Coherence - determine if the summary flows logically with smooth transitions. Evaluate based on: 1. Logical structure, 2. Smooth transitions between ideas, 3. Clear progression of thoughts.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.judge_model,
            threshold=config.ThresholdConfig.min_coherence,
            async_mode=False
        )
        metrics.append(coherence_metric)

        return metrics

    def evaluate_single(self, test_case: LLMTestCase, 
                       metrics: Optional[List[BaseMetric]] = None) -> Dict[str, Any]:
        """
        Evaluate a single test case

        Args:
            test_case: DeepEval test case
            metrics: List of metrics to evaluate (uses defaults if None)

        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = self.create_metrics()

        # Run evaluation
        results = evaluate(
            test_cases=[test_case],
            metrics=metrics
        )

        test_results_list = getattr(results, 'test_results', results)

        scores = {}
        if test_results_list and len(test_results_list) > 0:
            test_result = test_results_list[0]
            for metric_data in test_result.metrics_data:
                scores[metric_data.name] = {
                    'score': getattr(metric_data, 'score', 0.0),
                    'success': getattr(metric_data, 'is_successful', getattr(metric_data, 'success', False)),
                    'reason': getattr(metric_data, 'reason', None)
                }

        return scores

    def evaluate_batch(self, test_cases: List[LLMTestCase],
                      metrics: Optional[List[BaseMetric]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases in batch

        Args:
            test_cases: List of DeepEval test cases
            metrics: List of metrics to evaluate

        Returns:
            List of evaluation results
        """
        if metrics is None:
            metrics = self.create_metrics()

        # Run batch evaluation
        results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        # Compatibility for newer DeepEval versions
        test_results_list = getattr(results, 'test_results', results)

        # Extract results for each test case safely
        all_results = []
        for test_result in test_results_list:
            scores = {}
            for metric_data in test_result.metrics_data:
                scores[metric_data.name] = {
                    'score': getattr(metric_data, 'score', 0.0),
                    'success': getattr(metric_data, 'is_successful', getattr(metric_data, 'success', False)),
                    'reason': getattr(metric_data, 'reason', None)
                }
            all_results.append(scores)

        return all_results


def run_deepeval_evaluation(summaries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convenience function to run DeepEval evaluation

    Args:
        summaries: List of dicts with 'source', 'summary', and optional 'reference'

    Returns:
        List of evaluation results
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval is required")

    pipeline = DeepEvalPipeline(use_deepeval=True)

    # Create test cases
    test_cases = []
    for item in summaries:
        test_case = pipeline.create_test_case(
            input_text=item['source'],
            actual_output=item['summary'],
            expected_output=item.get('reference'),
            context=[item['source']],
            retrieval_context=[item['source']]
        )
        test_cases.append(test_case)

    # Evaluate
    results = pipeline.evaluate_batch(test_cases)

    return results


if __name__ == "__main__":
    # Example usage
    if DEEPEVAL_AVAILABLE:
        test_data = [
            {
                'source': "AI is transforming healthcare through diagnostic tools.",
                'summary': "AI improves healthcare diagnostics.",
                'reference': "Artificial intelligence is revolutionizing medical diagnostics."
            }
        ]

        results = run_deepeval_evaluation(test_data)
        print("Evaluation complete!")
        print(results)
    else:
        print("DeepEval not available. Install with: pip install deepeval")
