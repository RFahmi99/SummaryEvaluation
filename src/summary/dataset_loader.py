"""
Dataset loader module for various summarization datasets
"""
import random
from typing import List, Dict, Tuple
from datasets import load_dataset

class DatasetLoader:
    """Handles loading and sampling from various summarization datasets"""

    SUPPORTED_DATASETS = {
        'multi-news': ('Awesome075/multi_news_parquet', None),
        'wcep-10': ('ccdv/WCEP-10', None),  # WCEP alternative
        'cnndm': ('cnn_dailymail', '3.0.0'),
        'xsum': ('xsum', None),
        'wikihow': ('gursi26/wikihow-cleaned', None),  # Requires manual download (Local CSV files required)
    }

    def __init__(self, dataset_name: str, supported_datasets: dict, random_seed: int):
        """
        Initialize dataset loader

        Args:
            dataset_name: Name of the dataset to load
            random_seed: Seed for random sampling
        """
        self.dataset_name = dataset_name.lower()
        self.supported_datasets = supported_datasets
        self.random_seed = random_seed
        random.seed(random_seed)

        if self.dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported: {list(self.supported_datasets.keys())}")

    def load_dataset(self) -> List[Dict]:
        """Load the specified dataset"""
        dataset_info = self.supported_datasets[self.dataset_name]

        if dataset_info is None:
            raise NotImplementedError(
                f"Dataset {self.dataset_name} requires manual download. "
                f"Please provide the dataset files."
            )

        dataset_path, config = dataset_info
        print(f"Loading dataset: {self.dataset_name}...")

        try:
            if config:
                dataset = load_dataset(dataset_path, config, split='train')
            else:
                dataset = load_dataset(dataset_path, split='train')

            print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def sample_posts(self, dataset, num_posts: int) -> List[Dict]:
        """
        Randomly sample posts from dataset

        Args:
            dataset: Loaded dataset
            num_posts: Number of posts to sample

        Returns:
            List of sampled posts with source and reference
        """
        if num_posts > len(dataset):
            print(f"Warning: Requested {num_posts} posts but dataset has {len(dataset)}. "
                  f"Sampling all available.")
            num_posts = len(dataset)

        indices = random.sample(range(len(dataset)), num_posts)
        sampled_posts = []

        for idx in indices:
            post = dataset[idx]
            processed_post = self._process_post(post)
            sampled_posts.append(processed_post)

        print(f"Sampled {len(sampled_posts)} posts")
        return sampled_posts

    def _process_post(self, post: Dict) -> Dict:
        """Process post based on dataset format"""
        processed = {'source': '', 'reference': ''}

        if self.dataset_name == 'multi-news':
            processed['source'] = post.get('document', '')
            processed['reference'] = post.get('summary', '')

        elif self.dataset_name == 'cnndm':
            processed['source'] = post.get('article', '')
            processed['reference'] = post.get('highlights', '')

        elif self.dataset_name == 'xsum':
            processed['source'] = post.get('document', '')
            processed['reference'] = post.get('summary', '')

        elif self.dataset_name == 'wikihow':
            processed['source'] = post.get('text', '')
            processed['reference'] = post.get('headline', '')

        elif self.dataset_name == 'newsroom':
            processed['source'] = post.get('text', '')
            processed['reference'] = post.get('summary', '')

        elif self.dataset_name == 'nyt':
            processed['source'] = post.get('article', '')
            processed['reference'] = post.get('abstract', '')

        else:
            # Generic fallback
            processed['source'] = str(post.get('text', post.get('document', post.get('article', ''))))
            processed['reference'] = str(post.get('summary', post.get('headline', '')))

        return processed
