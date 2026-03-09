"""
Configuration file for the Summary Evaluation Pipeline
Customize model choices, thresholds, and evaluation settings
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple


@dataclass
class ResummarizationConfig:
    """Configuration for adaptive resummarization loops"""
    max_retries: int = 2  # Prevent infinite loops [cite: 280]
    retry_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'factual_consistency': 0.7,
        'relevance': 0.7
    })


@dataclass
class ModelConfig:
    """Configuration for evaluation models"""

    # Phase 2: Textual Quality Models
    similarity_model: str = "roberta-large"  # For BERTScore
    factual_model: str = "ufal/bge-m3-alignscore"  # AlignScore alternative
    judge_model: str = "llama3"  # Prometheus 2
    fluency_model: str = "meta-llama/Llama-3-8b"  # For perplexity

    # Phase 3: Safety Models
    safety_model: str = "meta-llama/LlamaGuard-7b"  # Or ShieldGemma

    # Inference settings
    device: str = "cuda"  # "cuda" or "cpu"
    use_fp16: bool = True
    batch_size: int = 8


@dataclass
class ThresholdConfig:
    """Quality and safety thresholds"""

    # Minimum acceptable scores (0-1 scale)
    # min_similarity: float = 0.7 
    # min_factual_consistency: float = 0.7
    # min_relevance: float = 0.6
    # min_coherence: float = 0.6
    # min_fluency: float = 0.7
    # min_fairness: float = 0.8
    min_similarity: float = 0.4 
    min_factual_consistency: float = 0.4
    min_relevance: float = 0.4
    min_coherence: float = 0.4
    min_fluency: float = 0.4
    min_fairness: float = 0.4

    # Safety thresholds
    max_toxicity: float = 0.3
    allow_pii: bool = False


@dataclass
class PipelineConfig:
    """Overall pipeline configuration"""

    # Enable/disable evaluation phases
    enable_similarity: bool = True
    enable_factual_consistency: bool = True
    enable_relevance: bool = True
    enable_coherence: bool = True
    enable_fluency: bool = True
    enable_fairness: bool = True
    enable_safety_gate: bool = True
    enable_pii_detection: bool = True
    enable_telemetry: bool = True

    # Logging and output
    log_level: str = "INFO"
    save_detailed_logs: bool = True
    output_dir: str = "./evaluation_results"

    # Performance settings
    max_workers: int = 8  # For parallel evaluation
    timeout_seconds: int = 1500  # Per-sample timeout

    resummarization: ResummarizationConfig = field(default_factory=ResummarizationConfig)


@dataclass
class GenerationConfig:
    """Configuration for the Summarization Generation Pipeline"""
    
    # Defaults
    default_seed: int = 42
    default_temperature: float = 0.5
    default_max_tokens: int = 32768
    
    # API Settings
    ollama_base_url: str = "http://localhost:11434"
    api_timeout_get: int = 5
    api_timeout_post: int = 1500
    
    # File Paths
    default_prompt_file: str = "src/summary/prompt.txt"
    fallback_prompt: str = """
{{ improvement_context }}

Please summarize the following text in under {{ max_word_count }} words. 
You MUST wrap your final summary exactly inside an ```article``` code block.

{{ sources }}
"""
    output_dir: str = "outputs"
    
    # Dataset Configurations
    supported_datasets: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=lambda: {
        'multi-news': ('multi_news', None),
        'wcep-10': ('webis/tldr-17', None),
        'cnndm': ('cnn_dailymail', '3.0.0'),
        'xsum': ('xsum', None),
        'wikihow': ('wikihow', 'all'),
    })


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_THRESHOLD_CONFIG = ThresholdConfig()
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()


def load_config(config_path: Optional[str] = None) -> tuple:
    """
    Load configuration from file or use defaults

    Args:
        config_path: Path to custom config JSON file

    Returns:
        Tuple of (ModelConfig, ThresholdConfig, PipelineConfig)
    """
    if config_path:
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        model_config = ModelConfig(**config_dict.get('models', {}))
        threshold_config = ThresholdConfig(**config_dict.get('thresholds', {}))
        pipeline_config = PipelineConfig(**config_dict.get('pipeline', {}))

        return model_config, threshold_config, pipeline_config

    return DEFAULT_MODEL_CONFIG, DEFAULT_THRESHOLD_CONFIG, DEFAULT_PIPELINE_CONFIG


def save_config(model_config: ModelConfig,
                threshold_config: ThresholdConfig,
                pipeline_config: PipelineConfig,
                output_path: str = "config.json"):
    """Save configuration to JSON file"""
    import json
    from dataclasses import asdict

    config_dict = {
        'models': asdict(model_config),
        'thresholds': asdict(threshold_config),
        'pipeline': asdict(pipeline_config)
    }

    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    # Generate default config file
    save_config(
        DEFAULT_MODEL_CONFIG,
        DEFAULT_THRESHOLD_CONFIG,
        DEFAULT_PIPELINE_CONFIG,
        "default_config.json"
    )
