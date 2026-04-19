# advanced_methods.py
"""
Advanced Market Basket Analysis methods for production systems.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class FPGrowthAnalyzer:
    """
    FP-Growth Algorithm Implementation
    
    Advantages over Apriori:
    - No candidate generation (faster)
    - Scans database only twice
    - Better for large datasets
    - Memory efficient with compressed representation
    """
    
    def __init__(self, min_support: float = 0.02):
        self.min_support = min_support
        self.frequent_itemsets = {}
        
    def fit(self, transactions: List[List[str]]):
        """Fit FP-Growth on transactions."""
        # This is a simplified implementation
        # For production, use: from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.preprocessing import TransactionEncoder
        
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)
        
        self.frequent_itemsets = fpgrowth(df, min_support=self.min_support, use_colnames=True)
        return self.frequent_itemsets


class CollaborativeFilteringRecommender:
    """
    Item-based Collaborative Filtering
    
    Use when you have:
    - Customer IDs (personalization possible)
    - Purchase history per customer
    - Need for personalized recommendations
    """
    
    def __init__(self):
        self.item_similarity = {}
        self.customer_purchases = defaultdict(set)
        
    def fit(self, transactions: List[Dict]):
        """
        Fit on transactions with customer IDs.
        
        Args:
            transactions: List of {'customer_id': str, 'items': List[str]}
        """
        # Build co-occurrence matrix
        item_counts = defaultdict(int)
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for txn in transactions:
            customer = txn.get('customer_id', 'anonymous')
            items = txn['items']
            
            self.customer_purchases[customer].update(items)
            
            for item in items:
                item_counts[item] += 1
                
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    co_occurrence[item1][item2] += 1
                    co_occurrence[item2][item1] += 1
        
        # Calculate cosine similarity
        for item1 in co_occurrence:
            self.item_similarity[item1] = {}
            for item2 in co_occurrence[item1]:
                similarity = co_occurrence[item1][item2] / np.sqrt(
                    item_counts[item1] * item_counts[item2]
                )
                self.item_similarity[item1][item2] = similarity
    
    def recommend(self, customer_id: str, current_cart: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        """Get personalized recommendations."""
        past_purchases = self.customer_purchases.get(customer_id, set())
        all_purchases = past_purchases.union(set(current_cart))
        
        scores = defaultdict(float)
        
        for item in current_cart:
            if item in self.item_similarity:
                for similar_item, similarity in self.item_similarity[item].items():
                    if similar_item not in all_purchases:
                        scores[similar_item] += similarity
        
        # Sort by score
        recommendations = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        return recommendations


class SequentialPatternMiner:
    """
    Sequential Pattern Mining for time-aware recommendations.
    
    Use when:
    - Order of purchases matters
    - Want to predict next purchase
    - Analyzing shopping journeys
    """
    
    def __init__(self, min_support: float = 0.01):
        self.min_support = min_support
        self.patterns = {}
        
    def fit(self, sequences: List[List[str]]):
        """
        Fit on purchase sequences.
        
        Args:
            sequences: List of item sequences ordered by time
        """
        # Count sequential patterns
        pattern_counts = defaultdict(int)
        total_sequences = len(sequences)
        
        for seq in sequences:
            # Generate subsequences
            for i in range(len(seq)):
                for j in range(i + 1, min(i + 4, len(seq) + 1)):  # Up to 3-item sequences
                    pattern = tuple(seq[i:j])
                    pattern_counts[pattern] += 1
        
        # Filter by support
        self.patterns = {
            pattern: count / total_sequences
            for pattern, count in pattern_counts.items()
            if count / total_sequences >= self.min_support
        }
        
        return self.patterns
    
    def predict_next(self, current_sequence: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Predict next item in sequence."""
        predictions = defaultdict(float)
        
        # Look for patterns that start with current sequence
        for pattern, support in self.patterns.items():
            if len(pattern) > len(current_sequence):
                # Check if current sequence is prefix of pattern
                if pattern[:len(current_sequence)] == tuple(current_sequence):
                    next_item = pattern[len(current_sequence)]
                    predictions[next_item] = max(predictions[next_item], support)
        
        return sorted(predictions.items(), key=lambda x: -x[1])[:top_n]


class ScalabilityStrategies:
    """
    Strategies for scaling MBA to large datasets.
    """
    
    @staticmethod
    def sampling_strategy(transactions: List, sample_size: int = 10000) -> List:
        """
        Random sampling for large datasets.
        
        For 1M+ transactions, analyze a representative sample.
        """
        import random
        
        if len(transactions) <= sample_size:
            return transactions
        
        return random.sample(transactions, sample_size)
    
    @staticmethod
    def time_windowing(transactions: List[Dict], window_days: int = 90) -> List:
        """
        Use recent transactions only (sliding window).
        
        Benefits:
        - Captures current trends
        - Reduces computation
        - More relevant recommendations
        """
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=window_days)
        
        return [
            txn for txn in transactions
            if txn.get('timestamp', datetime.now()) >= cutoff
        ]
    
    @staticmethod
    def parallel_apriori(transactions: List, n_jobs: int = 4):
        """
        Parallel processing for Apriori.
        
        Split transactions across cores for faster processing.
        """
        # Use joblib or multiprocessing
        from joblib import Parallel, delayed
        
        # Split transactions into chunks
        chunk_size = len(transactions) // n_jobs
        chunks = [
            transactions[i:i + chunk_size]
            for i in range(0, len(transactions), chunk_size)
        ]
        
        # Process in parallel (simplified example)
        # In practice, you'd need to merge results carefully
        return chunks
    
    @staticmethod
    def distributed_processing_setup():
        """
        For very large datasets, use distributed processing:
        
        1. Apache Spark MLlib - FP-Growth implementation
           ```python
           from pyspark.ml.fpm import FPGrowth
           
           fpGrowth = FPGrowth(itemsCol="items", minSupport=0.02, minConfidence=0.1)
           model = fpGrowth.fit(dataset)
           ```
        
        2. Dask - Parallel pandas on cluster
           ```python
           import dask.dataframe as dd
           
           ddf = dd.from_pandas(df, npartitions=10)
           # Process in parallel
           ```
        
        3. Redis for caching rules
           - Store association rules in Redis
           - Sub-millisecond recommendation lookups
        """
        pass


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Sample transactions
    transactions = [
        ["Milk", "Bread", "Butter"],
        ["Milk", "Bread"],
        ["Chips", "Soft Drinks"],
        ["Rice", "Dal", "Oil"],
        # ... more transactions
    ]
    
    # FP-Growth (faster for large datasets)
    print("=== FP-Growth Analysis ===")
    fpg = FPGrowthAnalyzer(min_support=0.1)
    # itemsets = fpg.fit(transactions)
    # print(itemsets)
    
    # Collaborative Filtering (for personalization)
    print("\n=== Collaborative Filtering ===")
    cf = CollaborativeFilteringRecommender()
    cf.fit([
        {"customer_id": "C001", "items": ["Milk", "Bread"]},
        {"customer_id": "C001", "items": ["Milk", "Butter"]},
        {"customer_id": "C002", "items": ["Rice", "Dal"]},
    ])
    recs = cf.recommend("C001", ["Tea"], top_n=3)
    print(f"Recommendations for C001 with Tea in cart: {recs}")
    
    # Sequential patterns
    print("\n=== Sequential Patterns ===")
    spm = SequentialPatternMiner(min_support=0.1)
    patterns = spm.fit([
        ["Milk", "Bread", "Butter"],
        ["Milk", "Bread", "Eggs"],
        ["Rice", "Dal", "Oil"],
    ])
    print(f"Patterns found: {len(patterns)}")
