"""
Main Summarization Pipeline
Author: Rayed Fahmi
Date: 2026-03-02

This pipeline loads datasets, samples posts, and summarizes them using Ollama LLMs.
"""
import argparse
import os
import sys
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from src.summary.dataset_loader import DatasetLoader
from src.summary.llm_handler import OllamaLLMHandler
from src.summary.output_handler import OutputHandler
from src.evaluation.evaluation import SummaryEvaluationPipeline
from config import DEFAULT_GENERATION_CONFIG, DEFAULT_PIPELINE_CONFIG

class GraphState(TypedDict):
    """Represents the state of our summarization workflow"""
    source_text: str
    reference: str
    dynamic_max_words: int
    attempt: int
    max_retries: int
    current_summary: str
    improvement_context: str
    passed_all_checks: bool
    feedback_logs: Dict[str, dict]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_time: float
    best_result: dict
    initial_passed_checks: Optional[bool]

class SummarizationPipeline:
    """Main pipeline for document summarization"""

    def __init__(self, config: dict, gen_config=DEFAULT_GENERATION_CONFIG):
        """
        Initialize the pipeline

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config
        self.gen_config = gen_config
        self.prompt = self._load_prompt()

        self.dataset_loader = DatasetLoader(
            config['dataset'],
            self.gen_config.supported_datasets,
            config['random_seed']
        )

        self.llm_handler = OllamaLLMHandler(
            config['llm_model'],
            config['temperature'],
            config['max_tokens'],
            base_url=self.gen_config.ollama_base_url,
            timeout_get=self.gen_config.api_timeout_get,
            timeout_post=self.gen_config.api_timeout_post
        )

        self.output_handler = OutputHandler(
            output_dir=self.gen_config.output_dir
        )

    def _load_prompt(self) -> str:
        """Load prompt from prompt.txt"""
        prompt_file = self.config.get('prompt_file', self.gen_config.default_prompt_file)

        if not os.path.exists(prompt_file):
            print(f"⚠ Warning: {prompt_file} not found. Using default prompt.")
            return self.gen_config.fallback_prompt

        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        print(f"✓ Loaded prompt from {prompt_file}")
        print(f"  Prompt preview: {prompt[:100]}...")
        return prompt

    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "=" * 80)
        print("SUMMARIZATION & EVALUATION PIPELINE")
        print("=" * 80)

        # Step 1: Load dataset
        print("\n[Step 1/4] Loading dataset...")
        dataset = self.dataset_loader.load_dataset()

        # Step 2: Sample posts
        print("\n[Step 2/4] Sampling posts...")
        sampled_posts = self.dataset_loader.sample_posts(
            dataset, 
            self.config['num_posts']
        )

        if not sampled_posts:
            print("❌ Error: No posts sampled. Exiting.")
            return None
        
        print("\n[Step 3/4] Intelligent Summarization Loop...")
        evaluator = SummaryEvaluationPipeline(enable_safety=True, enable_telemetry=False)
        results = []
        max_retries = self.config.get('max_retries', DEFAULT_PIPELINE_CONFIG.resummarization.max_retries)
        
        # 1. Define the Generator Node
        def generate_node(state: GraphState) -> GraphState:
            print(f"  Attempt {state['attempt'] + 1}...")
            
            final_summary, raw_resp, t_taken, p_tokens, c_tokens = self.llm_handler.summarize(
                self.prompt, state['source_text'], state['dynamic_max_words'], state['improvement_context']
            )
            
            # Update state with generation metrics
            state['current_summary'] = final_summary
            state['total_prompt_tokens'] += p_tokens
            state['total_completion_tokens'] += c_tokens
            state['total_time'] += t_taken
            state['attempt'] += 1
            return state

        # 2. Define the Evaluator Node
        def evaluate_node(state: GraphState) -> GraphState:
            summary = state['current_summary']
            source = state['source_text']
            reference = state['reference']
            
            # Run the specific Prometheus Judge prompts
            rel_eval = evaluator.textual_evaluator.evaluate_with_prometheus(summary, source, 'relevance')
            coh_eval = evaluator.textual_evaluator.evaluate_with_prometheus(summary, source, 'coherence')
            fact_eval = evaluator.textual_evaluator.evaluate_with_prometheus(summary, source, 'factual_consistency')
            
            # Run the missing evaluations to populate the CSV
            sim_score = evaluator.textual_evaluator.evaluate_similarity(summary, reference) if reference else None
            fluency_score = evaluator.textual_evaluator.evaluate_fluency(summary)
            fairness_score = evaluator.textual_evaluator.evaluate_fairness(summary, source)
            
            # Safety Gate checks
            safety_passed = True
            toxicity_score = 0.0
            if evaluator.enable_safety:
                safe_res = evaluator.safety_gate.evaluate(summary)
                safety_passed = safe_res['safety_passed']
                toxicity_score = safe_res['toxicity_score']

            feedbacks = {}
            if not rel_eval['passed']: feedbacks['relevance'] = rel_eval
            if not coh_eval['passed']: feedbacks['coherence'] = coh_eval
            if not fact_eval['passed']: feedbacks['factual_consistency'] = fact_eval
            if not safety_passed: feedbacks['safety'] = {'actionable_feedback': 'Summary failed safety checks.'}
            
            state['passed_all_checks'] = len(feedbacks) == 0
            state['feedback_logs'] = feedbacks
            
            # Record initial checks dynamically
            if state.get('initial_passed_checks') is None:
                state['initial_passed_checks'] = state['passed_all_checks']
            
            # Build improvement context for the next loop if it failed
            if not state['passed_all_checks']:
                print(f"  ✗ Failed checks. Issues: {list(feedbacks.keys())}")
                state['improvement_context'] = self.llm_handler.build_improvement_context(
                    feedbacks, previous_draft=summary
                )
            else:
                print("  ✓ Passed all checks!")

            # Save ALL current metrics to state
            state['best_result'] = {
                'source': source,
                'reference': reference,
                'summary': summary,
                'time_taken': state['total_time'],
                'total_attempts': state['attempt'],
                'prompt_tokens': state['total_prompt_tokens'],
                'completion_tokens': state['total_completion_tokens'],
                'total_tokens': state['total_prompt_tokens'] + state['total_completion_tokens'],
                'initial_passed_checks': state['initial_passed_checks'],
                'improvement_context_used': state['attempt'] > 1,
                'passed_all_checks': state['passed_all_checks'],
                'relevance_score': rel_eval['score'],
                'coherence_score': coh_eval['score'],
                'factual_consistency_score': fact_eval['score'],
                'similarity_score': sim_score,
                'fluency_score': fluency_score,
                'fairness_score': fairness_score,
                'safety_passed': safety_passed,
                'toxicity_score': toxicity_score,
                'failure_reasons': list(feedbacks.keys()) if not state['passed_all_checks'] else []
            }
            return state

        # 3. Define the Router
        def route_evaluation(state: GraphState):
            if state['passed_all_checks']:
                return END
            if state['attempt'] > state['max_retries']:
                print("  ⚠ Reached max retries. Ending loop.")
                return END
            return "generator"

        # 4. Compile the LangGraph
        workflow = StateGraph(GraphState)
        workflow.add_node("generator", generate_node)
        workflow.add_node("evaluator", evaluate_node)
        
        workflow.set_entry_point("generator")
        workflow.add_edge("generator", "evaluator")
        workflow.add_conditional_edges("evaluator", route_evaluation)
        
        app = workflow.compile()

        # Iterate through posts using the Graph
        for i, post in enumerate(sampled_posts, 1):
            print(f"\nProcessing post {i}/{len(sampled_posts)}...")
            raw_source_text = post['source']
            
            try:
                dataset_name = self.config['dataset'].lower()
                
                # Map datasets to their specific multi-document delimiters
                dataset_delimiters = {
                    'multi-news': "|||||",
                    'wcep-10': "</s>",  # ccdv/WCEP-10 default concatenation token
                }
                
                # Identify datasets that are strictly single-document
                single_doc_datasets = ['cnndm', 'xsum', 'wikihow']
                
                if dataset_name in single_doc_datasets:
                    # Do not split single-document datasets
                    articles = [raw_source_text.strip()]
                else:
                    # Attempt to split multi-document datasets
                    delimiter = dataset_delimiters.get(dataset_name)
                    
                    if delimiter and delimiter in raw_source_text:
                        articles = [art.strip() for art in raw_source_text.split(delimiter) if art.strip()]
                    else:
                        # Fallback to double newlines for other multi-doc datasets
                        # Only keep chunks that have a meaningful amount of words (>15)
                        articles = [art.strip() for art in raw_source_text.split('\n\n') if len(art.split()) > 15]
                        
                # Fallback to original text if splitting results in an empty list
                if not articles:
                    articles = [raw_source_text] 

                # Define the formatted text using the cleaned articles
                formatted_source_text = "\n\n".join(articles)

                # Calculate dynamic max_word_count based on the smallest article
                word_counts = [len(article.split()) for article in articles]
                min_article_words = min(word_counts)
                
                # Set dynamic max words (cap it at the smallest article's length)
                user_max_words = self.config.get('max_word_count', 750)
                dynamic_max_words = min(user_max_words, min_article_words)
                
                initial_state = GraphState(
                    source_text=formatted_source_text,
                    reference=post.get('reference', ''),
                    dynamic_max_words=dynamic_max_words,
                    attempt=0,
                    max_retries=max_retries,
                    current_summary="",
                    improvement_context="",
                    passed_all_checks=False,
                    feedback_logs={},
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    total_time=0.0,
                    best_result={},
                    initial_passed_checks=None
                )

                try:
                    # Run the LangGraph application
                    final_state = app.invoke(initial_state)
                    results.append(final_state['best_result'])
                except Exception as e:
                    print(f"  ❌ Error processing post {i}: {str(e)}")
                    results.append({'source': formatted_source_text, 'summary': f"ERROR: {str(e)}", 'passed_all_checks': False})

            # CATCH EXCEPTION TO PREVENT CRASHING
            except Exception as e:
                print(f"  ❌ Error processing post {i}: {str(e)}")
                # Append a failed placeholder so the pipeline records the failure and moves on
                results.append({
                    'source': raw_source_text,
                    'reference': post.get('reference', ''),
                    'summary': f"PIPELINE ERROR: {str(e)}",
                    'passed_all_checks': False,
                    'failure_reasons': [f"pipeline_fatal_error: {str(e)}"]
                })
                continue

        # Step 4: Save results
        print("\n[Step 4/4] Saving results to CSV...")
        additional_info = {
            'temperature': self.config['temperature'],
            'random_seed': self.config['random_seed'],
            'dataset': self.config['dataset'],
            'num_posts': self.config['num_posts']
        }

        output_path = self.output_handler.save_to_csv(
            results,
            self.config['llm_model'],
            self.config['max_tokens'],
            additional_info
        )

        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Output file: {output_path}")

        return output_path


def main():
    """Main entry point with CLI argument parsing"""
    gen_config = DEFAULT_GENERATION_CONFIG
    parser = argparse.ArgumentParser(
        description="Summarization Pipeline using Ollama LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --num_posts 10 --model llama2 --dataset cnndm
  python main.py --num_posts 5 --model mistral --temperature 0.5 --max_tokens 256 --dataset xsum --seed 123
  python main.py --help

Supported datasets:
  multi-news, wcep-10, cnndm, xsum, wikihow

Note: Make sure Ollama is running (ollama serve) and your model is pulled (ollama pull <model>)
        """
    )

    # Required arguments

    parser.add_argument('--num_posts', type=int, required=True,
                        help='Number of posts to sample and summarize')

    parser.add_argument('--model', '--llm', dest='llm_model', type=str, required=True,
                        help='Ollama model name (e.g., llama2, mistral, codellama)')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['multi-news', 'wcep-10', 'duc2004', 'tac2011', 
                                'cnndm', 'xsum', 'nyt', 'newsroom', 'wikihow', 'newshead'],
                        help='Dataset to use for sampling')

    # Optional arguments
    parser.add_argument('--seed', '--random_seed', dest='random_seed', type=int, default=gen_config.default_seed,
                        help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--temperature', type=float, default=gen_config.default_temperature,
                        help='LLM temperature (0.0-1.0, default: 0.7)')

    parser.add_argument('--max_tokens', type=int, default=gen_config.default_max_tokens,
                        help='Maximum tokens for LLM generation (default: 512)')

    parser.add_argument('--prompt_file', type=str, default=gen_config.default_prompt_file,
                        help='Path to prompt file (default: prompt.txt)')

    args = parser.parse_args()

    # Create configuration dictionary
    config = {
        'num_posts': args.num_posts,
        'random_seed': args.random_seed,
        'llm_model': args.llm_model,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'dataset': args.dataset,
        'prompt_file': args.prompt_file,
        'max_retries': DEFAULT_PIPELINE_CONFIG.resummarization.max_retries
    }

    # Print configuration
    print("\nPipeline Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)

    # Run pipeline
    try:
        pipeline = SummarizationPipeline(config)
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
