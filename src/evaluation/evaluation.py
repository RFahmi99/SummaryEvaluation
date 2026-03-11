"""
LLM Summary Evaluation Pipeline
Based on SOTA open-source tools and frameworks
"""

import re
import time
import torch
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from urllib import response
import config

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
        AutoModelForCausalLM,
        pipeline
    )
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
                logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
                
                self.analyzer = AnalyzerEngine(default_score_threshold=0.4) 
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Presidio: {e}")
                logger.info("Suggestion: Run 'python -m spacy download en_core_web_lg'")
                self.analyzer = None

    def _chunk_text(self, text: str, chunk_size: int = 300) -> list:
        """Helper to safely split text into word-based chunks."""
        words = str(text).split()
        if not words:
            return []
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def check_toxicity(self, text: str) -> tuple[bool, float]:
        """Check for toxicity, harm, and alignment issues using text chunking"""
        if not self.safety_classifier:
            logger.warning("Safety classifier not available, skipping toxicity check")
            return True, 0.0

        try:
            chunks = self._chunk_text(text, chunk_size=300)
            
            if not chunks:
                return True, 0.0
                
            is_safe_overall = True
            max_toxicity = 0.0

            for chunk in chunks:
                # Skip chunks that are too small to have meaningful context
                if len(chunk.split()) < 10:
                    continue
                    
                result = self.safety_classifier(chunk, truncation=True, max_length=512)
                label = result[0]['label'].lower()
                confidence = result[0]['score']

                is_safe = label not in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'unsafe']
                
                # If it's safe, baseline toxicity is 0.0 (or a very low number). 
                # Only use the confidence score directly if a toxic label is actively predicted.
                if is_safe:
                    toxicity_score = 0.0 
                else:
                    toxicity_score = confidence
                
                max_toxicity = max(max_toxicity, toxicity_score)
                
                if not is_safe:
                    is_safe_overall = False

            return is_safe_overall, max_toxicity

        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return True, 0.0

    def check_pii(self, text: str) -> tuple[bool, List[str]]:
        """Check for PII leakage using chunking to bypass Presidio's 10k character limit"""
        if not self.analyzer:
            logger.warning("Presidio not available, skipping PII check")
            return True, []

        try:
            # We can use larger chunks for Presidio since it doesn't use standard token limits,
            # but we must stay under its 10,000 character hard limit.
            chunks = self._chunk_text(text, chunk_size=1000)
            
            if not chunks:
                return True, []

            all_pii_types = set()

            for chunk in chunks:
                results = self.analyzer.analyze(
                    text=chunk,
                    language='en',
                    entities=[
                        "EMAIL_ADDRESS", "PHONE_NUMBER", 
                        "CREDIT_CARD", "US_SSN",
                    ]
                )
                
                # Aggregate any PII found in this chunk
                for result in results:
                    all_pii_types.add(result.entity_type)

            has_pii = len(all_pii_types) > 0

            if has_pii:
                logger.warning(f"PII detected: {list(all_pii_types)}")

            return not has_pii, list(all_pii_types)

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
            self.bert_scorer = BERTScorer(model_type=config.ModelConfig.similarity_model, lang="en", device=device_str)

            # Caps the default max length if the HF config left it infinitely large.
            if hasattr(self.bert_scorer, '_tokenizer'):
                tokenizer = self.bert_scorer._tokenizer
                if getattr(tokenizer, 'model_max_length', 0) > 100000:
                    tokenizer.model_max_length = 512

            logger.info("BERTScore model loaded")
        except Exception as e:
            logger.warning(f"Could not load BERTScore: {e}")
            self.bert_scorer = None

        # 2. Prometheus 2
        try:
            # Grab the model name dynamically from config
            prometheus_model_name = config.DEFAULT_MODEL_CONFIG.judge_model
            
            self.prometheus_tokenizer = AutoTokenizer.from_pretrained(
                prometheus_model_name
            )
            self.prometheus_model = AutoModelForCausalLM.from_pretrained(
                prometheus_model_name,
                dtype=torch.float16,           
                low_cpu_mem_usage=True,        
                device_map="auto",              
                # tie_word_embeddings=False
            )

            logger.info(f"Prometheus Judge ({prometheus_model_name}) loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Prometheus Judge: {e}")
            self.prometheus_model = None

        # 3. Fluency Model
        try:
            # Grab the model name dynamically from config
            fluency_model_name = config.DEFAULT_MODEL_CONFIG.fluency_model
            
            self.fluency_tokenizer = AutoTokenizer.from_pretrained(
                fluency_model_name
            )
            self.fluency_model = AutoModelForCausalLM.from_pretrained(
                fluency_model_name,
                dtype=torch.float16,           
                low_cpu_mem_usage=True,        
                device_map="auto"
            )

            logger.info(f"Fluency Model ({fluency_model_name}) loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Fluency Model: {e}")
            self.fluency_model = None
            self.fluency_tokenizer = None

    def _chunk_text(self, text: str, max_words: int = 300) -> list:
        """Helper method to split text into safe word-count chunks."""
        text = str(text)
        
        # Safeguard: Prevent massive contiguous strings (like base64 or long URLs) 
        # from causing tokenization overflow by breaking them every 100 characters.
        import re
        text = re.sub(r'(\S{100})', r'\1 ', text)
        
        words = text.split()
        if not words:
            return [""]
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def evaluate_similarity(self, generated: str, reference: str) -> dict:
        """BERTScore for similarity using cross-chunking for long documents."""
        if getattr(self, 'bert_scorer', None) is None:
            logger.warning("BERTScore not available")
            return {"score": 0.0, "passed": False, "actionable_feedback": "BERTScore not available"}

        try:
            # 1. Type guard for multi-reference datasets
            if isinstance(reference, list):
                reference = " ".join(str(r) for r in reference)
            
            gen_str = str(generated).strip()
            ref_str = str(reference).strip()
            
            if not gen_str or not ref_str:
                return {"score": 0.0, "passed": False, "actionable_feedback": "Empty generation or reference text."}

            # 2. Chunk both texts to stay well under the matrix and token limits
            chunk_size = 300
            gen_chunks = self._chunk_text(gen_str, chunk_size)
            ref_chunks = self._chunk_text(ref_str, chunk_size)

            # Safeguard: Enforce a strict character limit per chunk (~3000 chars is well within 512 tokens)
            gen_chunks = [chunk[:3000] for chunk in gen_chunks]
            ref_chunks = [chunk[:3000] for chunk in ref_chunks]

            # 3. Fast path: If both texts are short, do a standard comparison
            if len(gen_chunks) == 1 and len(ref_chunks) == 1:
                P, R, F1 = self.bert_scorer.score([gen_chunks[0]], [ref_chunks[0]], verbose=False)
                score = float(F1.item())
                threshold = getattr(config.ThresholdConfig, "min_similarity", 0.4)
                passed = score >= threshold
                return {
                    "score": score,
                    "passed": passed,
                    "actionable_feedback": "The summary accurately reflects the core concepts of the reference." if passed else f"Similarity score was {score:.2f} (Required: {threshold:.2f})."
                }

            # 4. Cross-chunk comparison for long texts
            chunk_f1_scores = []
            
            for g_chunk in gen_chunks:
                best_f1 = 0.0
                # Compare the generated chunk against every reference chunk
                for r_chunk in ref_chunks:
                    _, _, F1 = self.bert_scorer.score([g_chunk], [r_chunk], verbose=False)
                    score = float(F1.item())
                    if score > best_f1:
                        best_f1 = score
                
                # Keep the best match for this specific generated chunk
                chunk_f1_scores.append(best_f1)
                
            # 5. Average the best matching scores to get the overall similarity
            overall_f1 = sum(chunk_f1_scores) / len(chunk_f1_scores) if chunk_f1_scores else 0.0
            
            threshold = getattr(config.ThresholdConfig, "min_similarity", 0.4)
            passed = overall_f1 >= threshold
            
            if passed:
                feedback = "The summary accurately reflects the core concepts of the reference."
            else:
                feedback = f"Similarity score was {overall_f1:.2f} (Required: {threshold:.2f}). The summary deviates significantly from the expected reference. Ensure all main entities and concepts from the source are present."

            return {
                "score": overall_f1,
                "passed": passed,
                "actionable_feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Fluency evaluation failed: {e}")
            return {"score": 0.0, "passed": False, "actionable_feedback": f"Fluency evaluation failed: {e}"}

    def evaluate_with_prometheus(self, summary: str, source: str, dimension: str) -> dict:
        """Use Prometheus 2 to generate strict JSON feedback for Relevancy, Coherence, and Factual Consistency"""
        if not self.prometheus_model:
            logger.warning("Prometheus 2 not available")
            return {"score": 0.0, "passed": False, "actionable_feedback": "Prometheus model unavailable."}

        # Safeguard: Truncate source text to a reasonable length to prevent tokenization overflow.
        safe_source = " ".join(source.split()[:4000]) # roughly 4000 words (~6000 tokens)

        # Define custom prompts that force a strict JSON output
        prompts = {
            'relevance': f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

###The instruction to evaluate:
Evaluate the relevance of the following summary based on the original source text.
Source text: {safe_source}

###Response to evaluate:
{summary}

###Score Rubric:
How relevant is the summary to the source text?
Score 1: The summary is completely irrelevant, misses all essential information, or focuses entirely on trivial, off-topic details.
Score 2: The summary is poorly relevant; it misses the main core messages of the source but manages to capture a few minor or tangential details.
Score 3: The summary is moderately relevant; it captures some essential information but omits other key points, or it includes a significant amount of unnecessary fluff.
Score 4: The summary is highly relevant; it captures almost all of the essential information with very little irrelevant information or filler.
Score 5: The summary is perfectly relevant; it concisely captures all essential information from the source accurately without any extraneous or off-topic details.

###Feedback:""",
            
            'coherence': f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

###The instruction to evaluate:
Evaluate the coherence and logical flow of the following summary. Consider how well the sentences connect and whether the ideas progress in a structured, readable manner.

###Response to evaluate:
{summary}

###Score Rubric:
How coherent and logical is the summary?
Score 1: The summary completely lacks logical flow, is highly disjointed, and is extremely difficult to read or understand.
Score 2: The summary has severe structural issues. Sentences feel disconnected or randomly placed, making the core ideas hard to follow.
Score 3: The summary is moderately coherent but contains noticeable awkward transitions, structural jumps, or abrupt topic changes that disrupt the reading experience.
Score 4: The summary is highly coherent and mostly easy to read, with only very minor or infrequent issues in transitions or sentence flow.
Score 5: The summary flows perfectly with excellent readability, exceptionally smooth transitions, and a highly logical, natural progression of ideas.

###Feedback:""",

            'factual_consistency': f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

###The instruction to evaluate:
Evaluate the factual consistency of the following summary based on the provided source text.
Source text: {safe_source}

###Response to evaluate:
{summary}

###Score Rubric:
How factually consistent is the summary with the source text?
Score 1: The summary contains severe hallucinations, contradicts the source text directly, or fabricates primary entities/events.
Score 2: The summary contains major factual errors that change the core meaning of the source text.
Score 3: The summary is mostly faithful but contains minor factual inaccuracies, wrong numbers, or attributes actions to the wrong minor entities.
Score 4: The summary is faithful to the source text, with only negligible omissions of context that do not alter the truthfulness.
Score 5: The summary is perfectly faithful to the source text without any invented details, contradictions, or missing critical context.

###Feedback:""",

            'fairness': f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"

###The instruction to evaluate:
Evaluate the following summary for fairness, neutrality, and absence of bias based on the original source text.
Source text: {safe_source}

###Response to evaluate:
{summary}

###Score Rubric:
How fair, objective, and unbiased is the summary compared to the source text?
Score 1: The summary introduces severe bias, alters the neutral tone of the source, or unfairly misrepresents/excludes primary viewpoints present in the source.
Score 2: The summary contains major bias issues, uses a noticeably slanted tone, or heavily favors one side of an issue over another compared to the source.
Score 3: The summary is mostly neutral but contains subtle biases, loaded language, or a slight imbalance in how differing viewpoints are represented.
Score 4: The summary is highly objective and fair, with only negligible or highly debatable instances of non-neutral phrasing.
Score 5: The summary is perfectly objective, completely free of introduced bias, and represents the original text fairly, neutrally, and proportionately.

###Feedback:"""
        }

        try:
            prompt = prompts.get(dimension, prompts['relevance'])
            inputs = self.prometheus_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.prometheus_model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )

            input_length = inputs["input_ids"].shape[1]
            response_text = self.prometheus_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

            import re
            
            score = 0.0
            feedback = "No feedback provided."

            # Parse standard Prometheus format: "Feedback: <text> [RESULT] <score>"
            if "[RESULT]" in response_text:
                parts = response_text.split("[RESULT]")
                
                # 1. Extract and clean the Feedback
                raw_feedback = parts[0].strip()
                # Remove the leading "Feedback:" string if the model included it
                if raw_feedback.startswith("Feedback:"):
                    feedback = raw_feedback[len("Feedback:"):].strip()
                else:
                    feedback = raw_feedback
                
                # 2. Extract and normalize the Score
                score_text = parts[-1].strip()
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", score_text)
                if match:
                    # Normalize the 1-5 score to a 0.0-1.0 scale
                    score = float(match.group(1)) / 5.0
            else:
                return {"score": 0.0, "passed": False, "actionable_feedback": f"Failed to parse output. Missing [RESULT] tag. Raw output: {response_text}"}

            # Fetch threshold safely
            threshold = getattr(config.ThresholdConfig, f"min_{dimension}", 0.4)
            passed = score >= threshold
            
            return {
                "score": score,
                "passed": passed,
                "actionable_feedback": feedback
            }

        except Exception as e:
            logger.error(f"Prometheus evaluation failed for {dimension}: {e}")
            return {"score": 0.0, "passed": False, "actionable_feedback": f"Evaluation error: {str(e)}"}

    def evaluate_fluency(self, text: str) -> float:
        """
        Calculate perplexity for fluency using the already-loaded causal LM.
        Returns a normalized score between 0 and 1 (higher is more fluent).
        """
        # Pull the dedicated fluency model instead of the Prometheus judge
        eval_model = getattr(self, 'fluency_model', None)
        eval_tokenizer = getattr(self, 'fluency_tokenizer', None)
        
        if not eval_model or not eval_tokenizer:
            logger.warning("Model for fluency not available, returning default score")
            return {"score": 0.0, "passed": False, "actionable_feedback": "Fluency model not available"}

        try:
            encodings = eval_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=8192
            )
            input_ids = encodings.input_ids.to(eval_model.device)

            with torch.no_grad():
                outputs = eval_model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                
            perplexity = torch.exp(loss).item()
            
            # Normalize using inverse decay. 
            # Adjust 'baseline_p' based on your specific causal LM's average loss.
            baseline_p = 15.0 
            normalized_score = max(0.0, min(1.0, baseline_p / (baseline_p + perplexity - 1.0)))
            
            # Fetch threshold safely
            threshold = getattr(config.ThresholdConfig, "min_fluency", 0.4)
            passed = normalized_score >= threshold
            
            if passed:
                feedback = "The summary is fluent and natural." 
            else:
                feedback = f"Fluency score was {normalized_score:.2f} (Required: {threshold:.2f}). The summary reads unnaturally or contains structural issues. Improve sentence flow and grammar."            
            
            return {
                "score": normalized_score,
                "passed": passed,
                "actionable_feedback": feedback
            }

        except Exception as e:
            logger.error(f"Fluency evaluation failed: {e}")
            return {"score": 0.0, "passed": False, "actionable_feedback": f"Fluency evaluation failed: {e}"}

    def evaluate_all(self, summary: str, source: str, 
                     reference: Optional[str] = None) -> Dict[str, Any]:
        """Run all textual quality evaluations"""
        results = {}

        # Similarity (if reference available)
        if reference:
            results['similarity_score'] = self.evaluate_similarity(summary, reference)

        # Factual consistency
        fact_score, fact_feedback = self.evaluate_with_prometheus(
            summary, source, 'factual_consistency'
        )
        results['factual_consistency_score'] = fact_score
        results['factual_consistency_feedback'] = fact_feedback

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

        # Fairness
        fairness_eval = self.evaluate_with_prometheus(
            summary, source, 'fairness'
        )
        results['fairness_score'] = fairness_eval['score']
        results['fairness_feedback'] = fairness_eval['actionable_feedback']

        # Fluency
        results['fluency_score'] = self.evaluate_fluency(summary)

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

        # Phase 1: Orchestration & Text Quality
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

    def evaluate_summary(self, summary: str, source_text: str) -> EvaluationResult:
        """A simplified evaluation for offline testing (without LangGraph)"""
        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            source_text=source_text,
        )
        
        # 1. Safety Checks
        if self.enable_safety:
            safety_results = self.safety_gate.evaluate(summary)
            result.safety_passed = safety_results['safety_passed']
            if not result.safety_passed:
                result.passed_all_checks = False
                result.failure_reasons.append("Failed safety checks")
                return result

        # 2. Prometheus Quality Checks
        rel_eval = self.textual_evaluator.evaluate_with_prometheus(summary, source_text, 'relevance')
        coh_eval = self.textual_evaluator.evaluate_with_prometheus(summary, source_text, 'coherence')
        fact_eval = self.textual_evaluator.evaluate_with_prometheus(summary, source_text, 'factual_consistency')
        fairness_eval = self.textual_evaluator.evaluate_with_prometheus(summary, source_text, 'fairness')

        result.relevance_score = rel_eval['score']
        result.coherence_score = coh_eval['score']
        result.factual_consistency_score = fact_eval['score']
        result.fairness_score = fairness_eval['score']
        
        result.passed_all_checks = rel_eval['passed'] and coh_eval['passed'] and fact_eval['passed'] and fairness_eval['passed']
        
        if not result.passed_all_checks:
            if not rel_eval['passed']: result.failure_reasons.append("Relevance Failed")
            if not coh_eval['passed']: result.failure_reasons.append("Coherence Failed")
            if not fact_eval['passed']: result.failure_reasons.append("Factual Consistency Failed")
            if not fairness_eval['passed']: result.failure_reasons.append("Fairness Failed")

        return result

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
