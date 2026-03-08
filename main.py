"""
Main Summarization Pipeline
Author: Rayed Fahmi
Date: 2026-03-02

This pipeline loads datasets, samples posts, and summarizes them using Ollama LLMs.
"""
import argparse
import os
import sys

from src.summary.dataset_loader import DatasetLoader
from src.summary.llm_handler import OllamaLLMHandler
from src.summary.output_handler import OutputHandler
from src.evaluation.evaluation import SummaryEvaluationPipeline
from config import DEFAULT_GENERATION_CONFIG

os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "1000"

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
        
        print("\n[Step 3/4] Adaptive Summarization & Evaluation Loop...")
        
        evaluator = SummaryEvaluationPipeline(enable_safety=True, enable_telemetry=False)
        results = []
        max_retries = self.config.get('max_retries', 2)
        
        for i, post in enumerate(sampled_posts, 1):
            print(f"\nProcessing post {i}/{len(sampled_posts)}...")
            raw_source_text = post['source']

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

            # Calculate dynamic max_word_count based on the smallest article
            word_counts = [len(article.split()) for article in articles]
            min_article_words = min(word_counts)
            
            # Set dynamic max words (cap it at the smallest article's length)
            user_max_words = self.config.get('max_word_count', 750)
            dynamic_max_words = min(user_max_words, min_article_words)
            
            # Only reformat with "--- Article X ---" headers if it's actually a multi-doc dataset
            if len(articles) > 1:
                formatted_source_text = "\n\n".join([f"--- Article {idx+1} ---\n{art}" for idx, art in enumerate(articles)])
            else:
                formatted_source_text = articles[0]
            
            print(f"  Detected {len(articles)} article(s). Dynamic max_words set to: {dynamic_max_words}")
            
            attempt = 0
            passed_all_checks = False
            improvement_context = ""
            best_result = None
            
            # Aggregate counters
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_time = 0.0
            initial_passed = None

            # Wrap generation and evaluation in a while loop
            while attempt <= max_retries and not passed_all_checks:
                print(f"  Attempt {attempt + 1}...")
                
                # Attempt 1: Call with empty context. Retries: Use built context
                final_summary, raw_response, time_taken, prompt_tokens, comp_tokens = self.llm_handler.summarize(
                    self.prompt, formatted_source_text, dynamic_max_words, improvement_context
                )
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += comp_tokens
                total_time += time_taken
                
                # Evaluation
                eval_result = evaluator.evaluate_summary(
                    summary=final_summary, source_text=formatted_source_text, reference_summary=post.get('reference', None)
                )
                
                passed_all_checks = eval_result.passed_all_checks
                if attempt == 0:
                    initial_passed = passed_all_checks
                    
                # Store the best/latest iteration
                best_result = {
                    'source': formatted_source_text,
                    'reference': post.get('reference', ''),
                    'summary': final_summary,
                    'time_taken': total_time,
                    'prompt_tokens': total_prompt_tokens,
                    'completion_tokens': total_completion_tokens,
                    'total_tokens': total_prompt_tokens + total_completion_tokens,
                    'total_attempts': attempt + 1,
                    'initial_passed_checks': initial_passed,
                    'improvement_context_used': attempt > 0,
                    'passed_all_checks': passed_all_checks,
                    'similarity_score': eval_result.similarity_score,
                    'factual_consistency_score': eval_result.factual_consistency_score,
                    'relevance_score': eval_result.relevance_score,
                    'coherence_score': eval_result.coherence_score,
                    'fluency_score': eval_result.fluency_score,
                    'fairness_score': eval_result.fairness_score,
                    'safety_passed': eval_result.safety_passed,
                    'failure_reasons': eval_result.failure_reasons
                }
                
                # Check condition
                if passed_all_checks:
                    print("  ✓ Passed all checks!")
                    break
                else:
                    print(f"  ✗ Failed checks. Retrying... Reasons: {eval_result.failure_reasons}")
                    attempt += 1
                    if attempt <= max_retries:
                        improvement_context = self.llm_handler.build_improvement_context(
                            eval_result.feedback_logs,
                            previous_draft=final_summary
                        )

            # Handle ultimate failures
            if not passed_all_checks:
                print("  ⚠ Reached max retries. Flagging for human review.")
                best_result['failure_reasons'].append("requires_human_review")
                
            results.append(best_result)

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
