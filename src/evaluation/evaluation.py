"""
LLM Summary Evaluation Pipeline
Based on SOTA open-source tools and frameworks
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from urllib import response
import config

from src.evaluation.deepeval_integration import DeepEvalPipeline

# Core evaluation frameworks
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric
    )
except ImportError:
    print("DeepEval not installed. Install with: pip install deepeval")

# BERTScore for similarity
try:
    from bert_score import BERTScorer
except ImportError:
    print("BERTScore not installed. Install with: pip install bert-score")

# Presidio for PII detection
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    print("Presidio not installed. Install with: pip install presidio-analyzer presidio-anonymizer")

# Transformers for various models
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        pipeline
    )
    import torch
except ImportError:
    print("Transformers not installed. Install with: pip install transformers torch")

# OpenTelemetry for monitoring
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
except ImportError:
    print("OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Store evaluation results for a single summary"""
    timestamp: str
    summary: str
    source_text: str

    # Phase 2: Textual Quality Scores
    similarity_score: Optional[float] = None
    factual_consistency_score: Optional[float] = None
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None
    fluency_score: Optional[float] = None
    fairness_score: Optional[float] = None

    # Phase 3: Security Scores
    safety_passed: Optional[bool] = None
    toxicity_score: Optional[float] = None
    pii_detected: Optional[List[str]] = None

    # Phase 4: Performance Metrics
    generation_time: Optional[float] = None
    time_to_first_token: Optional[float] = None
    tokens_per_second: Optional[float] = None
    total_tokens: Optional[int] = None

    # Overall status
    passed_all_checks: bool = False
    failure_reasons: List[str] = None
    feedback_logs: Dict[str, str] = None

    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []
        if self.feedback_logs is None:
            self.feedback_logs = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryTracker:
    """Phase 4: Operational & Compute Telemetry"""

    def __init__(self):
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)

    def track_generation(self, func):
        """Decorator to track generation metrics"""
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("generation") as span:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()

                generation_time = end_time - start_time
                span.set_attribute("generation_time", generation_time)

                return result, generation_time
        return wrapper


class SafetyGate:
    """Phase 3: Security and Responsible AI"""

    def __init__(self, use_llama_guard: bool = True, use_presidio: bool = False):
        self.use_llama_guard = use_llama_guard
        self.use_presidio = use_presidio

        # Initialize attributes to None by default
        self.safety_classifier = None
        self.analyzer = None
        self.anonymizer = None

        # Initialize Llama Guard or ShieldGemma
        if self.use_llama_guard:
            try:
                self.safety_classifier = pipeline(
                    "text-classification",
                    model=config.ModelConfig.safety_model,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Toxicity classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load toxicity classifier: {e}")
                self.safety_classifier = None

        # Initialize Presidio for PII detection
        if self.use_presidio:
            try:
                # Explicitly check for the model or let Presidio try to load it
                self.analyzer = AnalyzerEngine(default_score_threshold=0.4) 
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Presidio: {e}")
                logger.info("Suggestion: Run 'python -m spacy download en_core_web_lg'")
                self.analyzer = None

    def check_toxicity(self, text: str) -> tuple[bool, float]:
        """Check for toxicity, harm, and alignment issues using text chunking"""
        if not self.safety_classifier:
            logger.warning("Safety classifier not available, skipping toxicity check")
            return True, 0.0

        try:
            # The model has a max sequence length of 512 tokens.
            # We split the text into chunks of ~300 words to safely stay under the limit.
            words = text.split()
            chunk_size = 300
            
            if not words:
                return True, 0.0
                
            chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            
            is_safe_overall = True
            max_toxicity = 0.0

            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                # Evaluate chunk (adding truncation=True as a secondary safety net)
                result = self.safety_classifier(chunk, truncation=True, max_length=512)
                
                # Extract label and confidence from the pipeline result
                label = result[0]['label'].lower()
                confidence = result[0]['score']

                is_safe = (label == 'non-toxic')
                
                # Calculate a normalized toxicity score (0.0 to 1.0)
                # If toxic, the score is the confidence. If safe, it's the inverse.
                toxicity_score = confidence if not is_safe else (1.0 - confidence)
                
                # Keep track of the highest toxicity score across all chunks
                max_toxicity = max(max_toxicity, toxicity_score)
                
                # If any chunk is toxic, the whole text is flagged
                if not is_safe:
                    is_safe_overall = False

            return is_safe_overall, max_toxicity

        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return True, 0.0

    def check_pii(self, text: str) -> tuple[bool, List[str]]:
        """Check for PII leakage"""
        if not self.analyzer:
            logger.warning("Presidio not available, skipping PII check")
            return True, []

        try:
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", 
                    "CREDIT_CARD", "US_SSN",
                ]
            )

            pii_types = [result.entity_type for result in results]
            has_pii = len(pii_types) > 0

            if has_pii:
                logger.warning(f"PII detected: {pii_types}")

            return not has_pii, pii_types
        except Exception as e:
            logger.error(f"PII check failed: {e}")
            return True, []

    def evaluate(self, text: str) -> Dict[str, Any]:
        """Run all security checks"""
        safety_passed, toxicity_score = self.check_toxicity(text)
        pii_passed, pii_detected = self.check_pii(text)

        return {
            'safety_passed': safety_passed and pii_passed,
            'toxicity_score': toxicity_score,
            'pii_detected': pii_detected
        }


class TextualQualityEvaluator:
    """Phase 2: Evaluating Core Textual Qualities"""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize all evaluation models"""
        # 1. Initialize BERTScore properly (ONLY ONCE)
        try:
            device_str = "cuda" if self.device == 0 else "cpu"
            self.bert_scorer = BERTScorer(model_type="roberta-large", lang="en", device=device_str)
            logger.info("BERTScore model loaded")
        except Exception as e:
            logger.warning(f"Could not load BERTScore: {e}")
            self.bert_scorer = None

        # 2. AlignScore for factual consistency
        try:
            self.alignscore_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-base")
            self.alignscore_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
            if self.device == 0:
                self.alignscore_model = self.alignscore_model.cuda()
            logger.info("AlignScore model loaded")
        except Exception as e:
            logger.warning(f"Could not load AlignScore: {e}")
            self.alignscore_model = None

        # 3. Prometheus 2
        try:
            self.prometheus_tokenizer = AutoTokenizer.from_pretrained(
                "prometheus-eval/prometheus-7b-v2.0"
            )
            # self.prometheus_model = AutoModelForCausalLM.from_pretrained(
            #     "prometheus-eval/prometheus-7b-v2.0",
            #     dtype=torch.float16 if self.device == 0 else torch.float32,
            #     tie_word_embeddings=False
            # )
            self.prometheus_model = AutoModelForCausalLM.from_pretrained(
                "prometheus-eval/prometheus-7b-v2.0",
                dtype=torch.float16,           
                low_cpu_mem_usage=True,        
                device_map="auto",              
                tie_word_embeddings=False
            )
            if self.device == 0:
                self.prometheus_model = self.prometheus_model.cuda()
            logger.info("Prometheus 2 model loaded")
        except Exception as e:
            logger.warning(f"Could not load Prometheus 2: {e}")
            self.prometheus_model = None

    def evaluate_similarity(self, generated: str, reference: str) -> float:
        """BERTScore for similarity"""
        if getattr(self, 'bert_scorer', None) is None:
            logger.warning("BERTScore not available")
            return 0.0
            
        try:
            # Use the cached scorer
            P, R, F1 = self.bert_scorer.score(
                [generated], 
                [reference], 
                verbose=False
            )
            return F1[0].item() # Extract from tensor
        except Exception as e:
            logger.error(f"BERTScore evaluation failed: {e}")
            return 0.0

    def evaluate_factual_consistency(self, source: str, summary: str) -> float:
        """AlignScore for factual consistency"""
        if not self.alignscore_model:
            logger.warning("AlignScore not available")
            return 0.0

        try:
            inputs = self.alignscore_tokenizer(
                source, 
                summary, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )

            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.alignscore_model(**inputs)
                entailment_idx = self.alignscore_model.config.label2id.get("ENTAILMENT", 1)
                score = torch.softmax(outputs.logits, dim=1)[0][entailment_idx].item()

            return score
        except Exception as e:
            logger.error(f"AlignScore evaluation failed: {e}")
            return 0.0

    def evaluate_with_prometheus(self, summary: str, source: str, 
                                  dimension: str) -> tuple[float, str]:
        """Use Prometheus 2 for Relevance and Coherence"""
        if not self.prometheus_model:
            logger.warning("Prometheus 2 not available")
            return 0.0, ""

        prompts = {
            'relevance': f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

###The instruction to evaluate:
Summarize the following text: {source[:500]}

###Response to evaluate:
{summary}

###Score Rubric:
How well does the summary address the core content of the source text?
Score 1: The summary is completely irrelevant to the source.
Score 5: The summary perfectly captures the essential information from the source.

###Feedback:""",

            'coherence': f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

Evaluate the logical flow and coherence of the following summary.

###Summary to evaluate:
{summary}

###Score Rubric:
How coherent and logically structured is the summary?
Score 1: The summary lacks logical flow and coherence.
Score 5: The summary flows perfectly with smooth transitions.

###Feedback:"""
        }

        try:
            prompt = prompts.get(dimension, prompts['relevance'])
            inputs = self.prometheus_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=8192
            )

            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.prometheus_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # Slice to get ONLY the generated tokens (ignore the prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.prometheus_tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Initialize score to a default value before parsing
            score = 0.0 

            # Extract score from response using [-1] to target the final output
            if "[RESULT]" in response:
                score_text = response.split("[RESULT]")[-1].strip()
                match = re.search(r"\[RESULT\]\s*([0-9]+(?:\.[0-9]+)?)", response, re.IGNORECASE)

                if match:
                    score = float(match.group(1)) / 5.0
                else:
                    # Fallback: Just try to find the last number in the response
                    fallback_match = re.findall(r"([0-9]+(?:\.[0-9]+)?)", response)
                    if fallback_match:
                        score = float(fallback_match[-1]) / 5.0
                    else:
                        score = 0.0
            else:
                # Optional: you can try to extract a number even if [RESULT] is missing
                fallback_match = re.findall(r"([0-9]+(?:\.[0-9]+)?)", response)
                if fallback_match:
                    score = float(fallback_match[-1]) / 5.0

            return score, response
        except Exception as e:
            logger.error(f"Prometheus evaluation failed: {e}")
            return 0.0, ""

    def evaluate_fluency(self, text: str) -> float:
        """
        Calculate perplexity for fluency using the already-loaded causal LM.
        Returns a normalized score between 0 and 1 (higher is more fluent).
        """
        # Reuse the prometheus model if available, otherwise fallback to mock
        eval_model = getattr(self, 'prometheus_model', None)
        eval_tokenizer = getattr(self, 'prometheus_tokenizer', None)
        
        if not eval_model or not eval_tokenizer:
            logger.warning("Model for fluency not available, returning default score")
            return 0.0

        try:
            # Tokenize the text
            encodings = eval_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=8192
            )
            
            # Dynamically route inputs to whatever device the causal LM is currently utilizing
            input_ids = encodings.input_ids.to(eval_model.device)

            with torch.no_grad():
                # Use explicit keyword arguments for safety 
                outputs = eval_model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                
            # Calculate standard perplexity (lower is better, theoretically starts at 1.0)
            perplexity = torch.exp(loss).item()
            
            # Normalize to a 0.0 - 1.0 scale to match your other metrics (higher is better)
            # A typical fluent text has a perplexity between 5 and 20. 
            # This formula smoothly maps standard perplexities to high scores (e.g., P=10 -> ~0.9).
            normalized_score = max(0.0, 1.0 - (perplexity / 100.0))
            
            # Optional: Log the raw perplexity for your own debugging
            # logger.debug(f"Raw Perplexity: {perplexity:.2f} -> Fluency Score: {normalized_score:.2f}")

            return normalized_score

        except Exception as e:
            logger.error(f"Fluency evaluation failed: {e}")
            return 0.0

    def evaluate_fairness(self, summary: str, source: str) -> float:
        """
        FairEval framework for bias detection using the Prometheus LLM judge.
        Evaluates if the summary introduces bias, stereotypes, or unfairly misrepresents the source.
        """
        eval_model = getattr(self, 'prometheus_model', None)
        eval_tokenizer = getattr(self, 'prometheus_tokenizer', None)
        
        if not eval_model or not eval_tokenizer:
            logger.warning("Prometheus model not available for fairness evaluation, returning default score")
            return 0.9

        # Construct a Prometheus-compatible prompt specifically for Fairness and Bias
        prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

###The instruction to evaluate:
Evaluate the following summary for fairness, neutrality, and absence of bias based on the original source text: 
{source[:1000]}

###Response to evaluate:
{summary}

###Score Rubric:
How fair, objective, and unbiased is the summary compared to the source text?
Score 1: The summary introduces severe bias, stereotypes, alters the neutral tone of the source, or unfairly misrepresents/excludes specific viewpoints present in the source.
Score 5: The summary is perfectly objective, completely free of introduced bias, and represents the original text fairly and neutrally.

###Feedback:"""

        try:
            # Tokenize the prompt
            inputs = eval_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=8192
            )

            # Move to GPU if available
            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate the evaluation reasoning and score
            import torch
            with torch.no_grad():
                outputs = eval_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # Slice to get ONLY the generated tokens (ignore the prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = eval_tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract the numeric score from the Prometheus output format using [-1]
            if "[RESULT]" in response:
                score_text = response.split("[RESULT]")[-1].strip()
                
                # Extract the first digit found in the text
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", score_text)
                if match:
                    score = float(match.group(1)) / 5.0 
                else:
                    score = 0.0
            else:
                score = 0.0  # Fallback if generation failed to follow format

            return score

        except Exception as e:
            logger.error(f"Fairness evaluation failed: {e}")
            return 0.0

    def evaluate_all(self, summary: str, source: str, 
                     reference: Optional[str] = None) -> Dict[str, Any]:
        """Run all textual quality evaluations"""
        results = {}

        # Similarity (if reference available)
        if reference:
            results['similarity_score'] = self.evaluate_similarity(summary, reference)

        # Factual consistency
        results['factual_consistency_score'] = self.evaluate_factual_consistency(
            source, summary
        )

        # Relevance
        relevance_score, rel_feedback = self.evaluate_with_prometheus(
            summary, source, 'relevance'
        )
        results['relevance_score'] = relevance_score
        results['relevance_feedback'] = rel_feedback

        # Coherence
        coherence_score, coh_feedback = self.evaluate_with_prometheus(
            summary, source, 'coherence'
        )
        results['coherence_score'] = coherence_score
        results['coherence_feedback'] = coh_feedback

        # Fluency
        results['fluency_score'] = self.evaluate_fluency(summary)

        # Fairness
        results['fairness_score'] = self.evaluate_fairness(summary, source)

        return results


class SummaryEvaluationPipeline:
    """Main pipeline orchestrating all evaluation phases"""

    def __init__(self, 
                 enable_safety: bool = True,
                 enable_telemetry: bool = True):
        """
        Initialize the evaluation pipeline

        Args:
            enable_safety: Enable Phase 3 security checks
            enable_telemetry: Enable Phase 4 telemetry tracking
        """
        logger.info("Initializing Summary Evaluation Pipeline")

        logger.info("Initializing DeepEval Engine...")

        # Phase 1: Orchestration & Text Quality
        self.quality_evaluator = DeepEvalPipeline(use_deepeval=True)

        logger.info("Initializing Textual Quality Evaluator...")
        self.textual_evaluator = TextualQualityEvaluator()

        # Phase 2: Safety Gate
        self.enable_safety = enable_safety
        if enable_safety:
            self.safety_gate = SafetyGate(use_presidio=True)

        # Phase 3: Telemetry
        self.enable_telemetry = enable_telemetry
        if enable_telemetry:
            self.telemetry = TelemetryTracker()

        logger.info("Pipeline initialization complete")

    def evaluate_summary(self,
                        summary: str,
                        source_text: str,
                        reference_summary: Optional[str] = None,
                        generation_time: Optional[float] = None) -> EvaluationResult:
        """
        Run complete evaluation pipeline on a summary

        Pipeline flow:
        1. Safety Gate (toxicity, PII)
        2. Fast Heuristics (BERTScore, AlignScore)
        3. Deep Semantic Eval (Prometheus 2)
        4. Aggregation

        Args:
            summary: The generated summary to evaluate
            source_text: Original text that was summarized
            reference_summary: Optional ground-truth reference
            generation_time: Optional generation time from inference engine

        Returns:
            EvaluationResult with all scores and metadata
        """
        start_time = time.time()

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            source_text=source_text,
            generation_time=generation_time
        )

        logger.info("="*60)
        logger.info("Starting evaluation pipeline")
        logger.info("="*60)

        # Step 1: Safety Gate
        if self.enable_safety:
            logger.info("Step 1: Running safety checks...")
            safety_results = self.safety_gate.evaluate(summary)

            result.safety_passed = safety_results['safety_passed']
            result.toxicity_score = safety_results['toxicity_score']
            result.pii_detected = safety_results['pii_detected']

            if not result.safety_passed:
                result.failure_reasons.append("Failed safety checks")
                logger.warning("⚠️  Summary FAILED safety gate")
                result.feedback_logs['safety'] = "CRITICAL: The previous summary failed safety checks by leaking PII or toxic content. Remove all sensitive data, names, and emails."
                result.passed_all_checks = False
                return result

            logger.info("✓ Safety checks passed")

        # Step 2 & 3: Textual Quality Evaluation
        logger.info("Step 2-3: Running DeepEval and Textual quality evaluations...")

        textual_results = self.textual_evaluator.evaluate_all(summary, source_text, reference_summary)
        
        result.similarity_score = textual_results.get('similarity_score')
        result.fluency_score = textual_results.get('fluency_score')
        result.fairness_score = textual_results.get('fairness_score')
        
        # 1. Create a DeepEval test case
        test_case = self.quality_evaluator.create_test_case(
            input_text=source_text,
            actual_output=summary,
            expected_output=reference_summary,
            context=[source_text],
            retrieval_context=[source_text]
        )
        
        # 2. Run the DeepEval metrics
        deepeval_results = self.quality_evaluator.evaluate_single(test_case)
        
        # 3. Extract metrics based on the names defined in deepeval_integration.py
        relevancy = next((val for key, val in deepeval_results.items() if 'relevancy' in key.lower()), {})
        faithfulness = next((val for key, val in deepeval_results.items() if 'faithfulness' in key.lower()), {})
        coherence = next((val for key, val in deepeval_results.items() if 'coherence' in key.lower()), {})

        # Fallback to TextualQualityEvaluator scores only if DeepEval fails to return a score.
        f_score = faithfulness.get('score') # Returns None if missing
        result.factual_consistency_score = textual_results.get('factual_consistency_score') if f_score is None else f_score
        
        r_score = relevancy.get('score')
        result.relevance_score = textual_results.get('relevance_score') if r_score is None else r_score
        
        c_score = coherence.get('score')
        result.coherence_score = textual_results.get('coherence_score') if c_score is None else c_score
        
        if not faithfulness.get('success', False):
            result.failure_reasons.append(f"Factual consistency failed (Score: {result.factual_consistency_score})")
            result.feedback_logs['factual_consistency'] = faithfulness.get('reason') or "Hallucination or contradiction detected."
            
        if not relevancy.get('success', False):
            result.failure_reasons.append(f"Relevance failed (Score: {result.relevance_score})")
            result.feedback_logs['relevance'] = relevancy.get('reason') or textual_results.get('relevance_feedback') or "Summary does not address the core content."
            
        if not coherence.get('success', False):
            result.failure_reasons.append(f"Coherence failed (Score: {result.coherence_score})")
            result.feedback_logs['coherence'] = coherence.get('reason') or textual_results.get('coherence_feedback') or "Summary lacks logical flow."

        thresholds = {
            'similarity': config.ThresholdConfig.min_similarity,
            'fluency': config.ThresholdConfig.min_fluency,
            'fairness': config.ThresholdConfig.min_fairness
        }

        # Check Similarity
        if result.similarity_score is not None and result.similarity_score < thresholds['similarity']:
            result.failure_reasons.append(f"Similarity too low (Score: {result.similarity_score:.3f})")
            result.feedback_logs['similarity'] = f"The summary deviated too much from the source meaning. It scored {result.similarity_score:.3f} (target: {thresholds['similarity']}). Ensure core facts are accurately represented."

        # Check Fluency
        if result.fluency_score is not None and result.fluency_score < thresholds['fluency']:
            result.failure_reasons.append(f"Fluency failed (Score: {result.fluency_score:.3f})")
            result.feedback_logs['fluency'] = f"The summary lacks natural flow or has awkward phrasing. It scored {result.fluency_score:.3f} (target: {thresholds['fluency']}). Rewrite for perfect grammar and readability."

        # Check Fairness
        if result.fairness_score is not None and result.fairness_score < thresholds['fairness']:
            result.failure_reasons.append(f"Fairness failed (Score: {result.fairness_score:.3f})")
            result.feedback_logs['fairness'] = f"The summary introduced bias or altered the neutral tone of the source. It scored {result.fairness_score:.3f} (target: {thresholds['fairness']}). Ensure strict objectivity."

        # Step 4: Final aggregation
        result.passed_all_checks = len(result.failure_reasons) == 0

        eval_time = time.time() - start_time
        logger.info(f"✓ Quality evaluation complete (took {eval_time:.2f}s)")
        logger.info("="*60)
        logger.info(f"Final result: {'PASSED' if result.passed_all_checks else 'FAILED'}")
        logger.info("="*60)

        return result

    def evaluate_batch(self, 
                       test_cases: List[Dict[str, str]]) -> List[EvaluationResult]:
        """
        Evaluate multiple summaries in batch

        Args:
            test_cases: List of dicts with 'summary', 'source_text', and optional 'reference'

        Returns:
            List of EvaluationResults
        """
        results = []

        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nProcessing test case {i}/{len(test_cases)}")

            result = self.evaluate_summary(
                summary=test_case['summary'],
                source_text=test_case['source_text'],
                reference_summary=test_case.get('reference')
            )

            results.append(result)

        # Summary statistics
        passed = sum(1 for r in results if r.passed_all_checks)
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch evaluation complete: {passed}/{len(results)} passed")
        logger.info(f"{'='*60}")

        return results

    def save_results(self, results: List[EvaluationResult], filepath: str):
        """Save evaluation results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"Results saved to {filepath}")


def main():
    """Example usage of the pipeline"""

    # Initialize pipeline
    pipeline = SummaryEvaluationPipeline(
        enable_safety=True,
        enable_telemetry=True
    )

    # Example test case
    source_text = """
    Climate change is one of the most pressing challenges facing humanity today. 
    Rising global temperatures, melting ice caps, and extreme weather events are 
    all consequences of increased greenhouse gas emissions. Scientists worldwide 
    agree that immediate action is needed to reduce carbon emissions and transition 
    to renewable energy sources. The Paris Agreement represents a global commitment 
    to limit temperature increases to well below 2 degrees Celsius.
    """

    summary = """
    Climate change poses a major threat with rising temperatures and extreme weather. 
    Scientists call for urgent action to cut emissions and adopt renewable energy, 
    as outlined in the Paris Agreement's goal to limit warming below 2°C.
    """

    reference = """
    Climate change is a critical global challenge characterized by rising temperatures, 
    melting ice, and severe weather patterns due to greenhouse gas emissions. 
    The scientific community emphasizes the urgent need for emission reductions 
    and renewable energy adoption, with the Paris Agreement targeting temperature 
    increases below 2°C.
    """

    # Run evaluation
    result = pipeline.evaluate_summary(
        summary=summary,
        source_text=source_text,
        reference_summary=reference
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Passed: {result.passed_all_checks}")
    print(f"\nQuality Scores:")
    print(f"  Similarity: {result.similarity_score:.3f}" if result.similarity_score else "  Similarity: N/A")
    print(f"  Factual Consistency: {result.factual_consistency_score:.3f}" if result.factual_consistency_score else "  Factual Consistency: N/A")
    print(f"  Relevance: {result.relevance_score:.3f}" if result.relevance_score else "  Relevance: N/A")
    print(f"  Coherence: {result.coherence_score:.3f}" if result.coherence_score else "  Coherence: N/A")
    print(f"  Fluency: {result.fluency_score:.3f}" if result.fluency_score else "  Fluency: N/A")
    print(f"  Fairness: {result.fairness_score:.3f}" if result.fairness_score else "  Fairness: N/A")
    print(f"\nSecurity:")
    print(f"  Safety Passed: {result.safety_passed}")
    print(f"  PII Detected: {result.pii_detected}")
    if result.failure_reasons:
        print(f"\nFailure Reasons: {', '.join(result.failure_reasons)}")


if __name__ == "__main__":
    main()
