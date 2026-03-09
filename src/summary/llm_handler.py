"""
LLM handler module for Ollama models
"""
import re
import time
import requests
import json
from typing import Dict, Tuple
from jinja2 import Template

class OllamaLLMHandler:
    """Handles interactions with Ollama LLM"""

    def __init__(self, model_name: str, temperature: float, 
                 max_tokens: int, base_url: str, timeout_get: int, timeout_post: int):
        """
        Initialize Ollama LLM handler

        Args:
            model_name: Name of the Ollama model (e.g., 'llama2', 'mistral')
            temperature: Temperature for generation (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        self.timeout_get = timeout_get
        self.timeout_post = timeout_post

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_get)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                print(f"✓ Connected to Ollama. Available models: {available_models}")

                # Check if requested model is available
                if not any(self.model_name in model for model in available_models):
                    print(f"⚠ Warning: Model '{self.model_name}' not found in available models.")
                    print(f"   You may need to pull it first: ollama pull {self.model_name}")
            else:
                print(f"⚠ Warning: Could not fetch Ollama models (status {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Warning: Could not connect to Ollama at {self.base_url}")
            print(f"   Make sure Ollama is running: {e}")

    def build_improvement_context(self, feedback_logs: Dict[str, str], previous_draft: str = "") -> str:
        """Constructs a strict directive based on failed evaluation metrics"""
        if not feedback_logs:
            return ""
            
        context = "CRITICAL DIRECTIVE: You are revising a previous summary that failed quality checks.\n\n"
        
        if previous_draft:
            context += f"### Previous Failed Draft:\n{previous_draft}\n\n"
            
        context += "### Pinpointed Faults to Fix:\n"
        for i, (metric, feedback) in enumerate(feedback_logs.items(), 1):
            context += f"{i}. [{metric.upper()}]: {feedback}\n"
            
        context += "\nRewrite the summary completely. You MUST resolve the exact faults listed above while maintaining all other constraints. Do not repeat the mistakes of the previous draft."
        
        return context

    def summarize(self, prompt_template: str, source_text: str, max_word_count: int, improvement_context: str = "") -> Tuple[str, str, float, int, int]:
        """
        Summarize text using the LLM

        Args:
            prompt_template: The prompt template to use for summarization
            source_text: Source text to summarize
            max_word_count: Maximum word count for the summary

        Returns:
            Tuple of (summary, time_taken_seconds)
        """
        # Render the Jinja2 template with the required variables
        template = Template(prompt_template)
        full_prompt = template.render(
            sources=source_text,
            max_word_count=max_word_count,
            improvement_context=improvement_context, 
            grouping_context=""
        )

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        start_time = time.time()

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout_post
            )
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()
                
                # Extract token metrics from Ollama's response
                prompt_tokens = result.get('prompt_eval_count', 0)
                completion_tokens = result.get('eval_count', 0)
                
                # Extract just the final article block
                final_summary = self._extract_summary(raw_response)

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            
            if isinstance(e, ValueError) and "Failed to extract" in str(e):
                raise e
                
            return error_msg, error_msg, elapsed_time, 0, 0

    def _extract_summary(self, raw_response: str) -> str:
        """
        Extract the final summary from the LLM's raw response using multiple fallback strategies.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Strategy 1: Strict article block with optional newline after ```article
        pattern1 = r'```[ \t]*article[ \t]*\n?(.*?)```'
        match = re.search(pattern1, raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Strategy 2: "## Headline" marker (as instructed in the prompt)
        pattern2 = r'##\s*Headline\s*\n(.*?)(?=```|$)'
        match = re.search(pattern2, raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Strategy 3: Last code block of any language (often the final article)
        code_blocks = re.findall(r'```(?:\w*)\n(.*?)```', raw_response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        # Strategy 4: Desperate fallback – return the whole response
        logger.warning("No article block or headline found; using full response as summary.")
        return raw_response.strip()

    def batch_summarize(self, prompt: str, posts: list, max_word_count: int = 150) -> list:
        """
        Summarize multiple posts

        Args:
            prompt: The prompt template
            posts: List of post dictionaries with 'source' key
            max_word_count: Maximum word count for each summary

        Returns:
            List of results with summaries and timing
        """
        results = []
        total = len(posts)

        print(f"\nStarting summarization of {total} posts...")
        print(f"Model: {self.model_name} | Temperature: {self.temperature} | Max Tokens: {self.max_tokens}")
        print("-" * 80)

        for i, post in enumerate(posts, 1):
            print(f"\nProcessing post {i}/{total}...")
            source_text = post['source']

            final_summary, raw_response, time_taken, prompt_tokens, completion_tokens = self.summarize(prompt, source_text, max_word_count)

            result = {
                'source': source_text,
                'reference': post.get('reference', ''),
                'summary': final_summary,
                'raw_response': raw_response,
                'time_taken': time_taken,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
            results.append(result)

            print(f"  ✓ Completed in {time_taken:.2f} seconds")
            print(f"  ✓ Tokens used: {prompt_tokens} (prompt) + {completion_tokens} (completion)")
            print(f"  Summary preview: {final_summary[:100]}...")

        print("\n" + "=" * 80)
        print(f"✓ All {total} posts summarized successfully!")
        avg_time = sum(r['time_taken'] for r in results) / len(results)
        print(f"  Average time per post: {avg_time:.2f} seconds")

        return results
