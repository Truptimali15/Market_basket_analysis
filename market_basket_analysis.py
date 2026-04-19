# market_basket_analysis.py
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MarketBasketAnalyzer:
    """
    Complete Market Basket Analysis system using the Apriori algorithm.
    
    Key Concepts:
    -------------
    SUPPORT: How frequently an itemset appears in the dataset.
        Formula: Support(A) = (Transactions containing A) / (Total transactions)
        Example: If Milk appears in 70 of 200 transactions, Support(Milk) = 0.35 (35%)
    
    CONFIDENCE: How often items in Y appear in transactions that contain X.
        Formula: Confidence(X → Y) = Support(X ∪ Y) / Support(X)
        Example: If Milk→Bread has confidence 0.6, it means 60% of customers 
                 who buy Milk also buy Bread.
    
    LIFT: How much more likely Y is purchased when X is purchased, compared to 
          Y's general purchase rate.
        Formula: Lift(X → Y) = Confidence(X → Y) / Support(Y)
        Interpretation:
            - Lift > 1: Positive association (items are bought together)
            - Lift = 1: No association (independent)
            - Lift < 1: Negative association (items substitute each other)
    """
    
    def __init__(self, min_support: float = 0.02, min_confidence: float = 0.1):
        """
        Initialize the analyzer.
        
        Args:
            min_support: Minimum support threshold (0-1). Lower = more itemsets found.
            min_confidence: Minimum confidence for rules (0-1).
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.df_encoded = None
        self.frequent_itemsets = None
        self.rules = None
        self.item_frequencies = defaultdict(int)
        
    def load_transactions(self, transactions: List[List[str]]):
        """Load transactions from list of lists format."""
        self.transactions = transactions
        self._encode_transactions()
        self._calculate_item_frequencies()
        print(f"Loaded {len(transactions)} transactions")
        
    def load_from_csv(self, filepath: str):
        """Load transactions from CSV file."""
        df = pd.read_csv(filepath)
        
        if 'items' in df.columns:
            # Format: transaction_id, date, items (comma-separated)
            self.transactions = [
                row['items'].split(',') for _, row in df.iterrows()
            ]
        else:
            # One-hot encoded format
            item_columns = [col for col in df.columns if col != 'transaction_id']
            self.transactions = []
            for _, row in df.iterrows():
                items = [col for col in item_columns if row[col] == 1]
                self.transactions.append(items)
        
        self._encode_transactions()
        self._calculate_item_frequencies()
        print(f"Loaded {len(self.transactions)} transactions from {filepath}")
    
    def _encode_transactions(self):
        """One-hot encode transactions for Apriori algorithm."""
        te = TransactionEncoder()
        te_array = te.fit_transform(self.transactions)
        self.df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
    def _calculate_item_frequencies(self):
        """Calculate frequency of each item."""
        self.item_frequencies = defaultdict(int)
        for txn in self.transactions:
            for item in txn:
                self.item_frequencies[item] += 1
    
    def run_apriori(self) -> pd.DataFrame:
        """
        Run the Apriori algorithm to find frequent itemsets.
        
        The Apriori algorithm works by:
        1. Finding all items that meet minimum support (frequent 1-itemsets)
        2. Using these to generate candidate 2-itemsets
        3. Pruning candidates that don't meet minimum support
        4. Repeating for larger itemsets until no more frequent itemsets found
        
        Returns:
            DataFrame with frequent itemsets and their support values
        """
        print(f"\nRunning Apriori with min_support={self.min_support}...")
        
        self.frequent_itemsets = apriori(
            self.df_encoded,
            min_support=self.min_support,
            use_colnames=True,
            verbose=0
        )
        
        # Add itemset length
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        return self.frequent_itemsets
    
    def generate_rules(self, metric: str = "lift", min_threshold: float = 1.0) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            metric: Metric to use for filtering ('lift', 'confidence', 'support')
            min_threshold: Minimum value for the metric
            
        Returns:
            DataFrame with association rules
        """
        if self.frequent_itemsets is None:
            self.run_apriori()
        
        print(f"\nGenerating rules with min_{metric}={min_threshold}...")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # Filter by confidence
        self.rules = self.rules[self.rules['confidence'] >= self.min_confidence]
        
        # Sort by lift (most interesting rules first)
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        print(f"Generated {len(self.rules)} association rules")
        return self.rules
    
    def get_recommendations(self, cart_items: List[str], top_n: int = 5) -> List[Dict]:
        """
        Get product recommendations based on items in cart.
        
        Args:
            cart_items: List of items currently in cart
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended items with confidence and lift scores
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        cart_set = set(cart_items)
        recommendations = defaultdict(lambda: {'confidence': 0, 'lift': 0, 'support': 0})
        
        for _, rule in self.rules.iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            # Check if cart contains the antecedents
            if antecedents.issubset(cart_set):
                # Recommend items in consequents that aren't in cart
                for item in consequents:
                    if item not in cart_set:
                        # Keep highest confidence rule for each item
                        if rule['confidence'] > recommendations[item]['confidence']:
                            recommendations[item] = {
                                'confidence': rule['confidence'],
                                'lift': rule['lift'],
                                'support': rule['support']
                            }
        
        # Sort by confidence and return top N
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: (x[1]['confidence'], x[1]['lift']),
            reverse=True
        )
        
        return [
            {
                'item': item,
                'confidence': round(data['confidence'], 3),
                'lift': round(data['lift'], 3),
                'reason': f"Customers who bought {', '.join(cart_items)} also bought this"
            }
            for item, data in sorted_recs[:top_n]
        ]
    
    def get_top_itemsets(self, min_length: int = 2, top_n: int = 20) -> pd.DataFrame:
        """Get top frequent itemsets of specified minimum length."""
        if self.frequent_itemsets is None:
            self.run_apriori()
            
        filtered = self.frequent_itemsets[self.frequent_itemsets['length'] >= min_length]
        return filtered.nlargest(top_n, 'support')
    
    def get_item_statistics(self) -> pd.DataFrame:
        """Get statistics for individual items."""
        total_txns = len(self.transactions)
        
        stats = []
        for item, count in self.item_frequencies.items():
            stats.append({
                'item': item,
                'frequency': count,
                'support': count / total_txns,
                'percentage': f"{count / total_txns * 100:.1f}%"
            })
        
        return pd.DataFrame(stats).sort_values('frequency', ascending=False)
    
    def print_analysis_report(self):
        """Print a comprehensive analysis report."""
        print("\n" + "="*70)
        print("         MARKET BASKET ANALYSIS REPORT - SUDHIR SUPERSHOPY")
        print("="*70)
        
        # Dataset overview
        print("\n📊 DATASET OVERVIEW")
        print("-" * 40)
        print(f"Total Transactions: {len(self.transactions)}")
        total_items = sum(len(t) for t in self.transactions)
        print(f"Total Items Sold: {total_items}")
        print(f"Average Basket Size: {total_items / len(self.transactions):.2f}")
        print(f"Unique Products: {len(self.item_frequencies)}")
        
        # Top items
        print("\n🏆 TOP 10 BEST-SELLING ITEMS")
        print("-" * 40)
        item_stats = self.get_item_statistics()
        for i, row in item_stats.head(10).iterrows():
            print(f"  {row['item']:20} | {row['frequency']:4} sales ({row['percentage']})")
        
        # Frequent itemsets
        if self.frequent_itemsets is not None:
            print("\n🔗 TOP FREQUENT ITEM COMBINATIONS")
            print("-" * 40)
            top_pairs = self.get_top_itemsets(min_length=2, top_n=10)
            for _, row in top_pairs.iterrows():
                items = ', '.join(row['itemsets'])
                print(f"  {items:40} | Support: {row['support']:.3f}")
        
        # Association rules
        if self.rules is not None and len(self.rules) > 0:
            print("\n📈 TOP ASSOCIATION RULES")
            print("-" * 70)
            print(f"{'IF (Antecedent)':<25} {'THEN (Consequent)':<20} {'Conf':>8} {'Lift':>8}")
            print("-" * 70)
            
            for _, rule in self.rules.head(15).iterrows():
                ant = ', '.join(rule['antecedents'])
                con = ', '.join(rule['consequents'])
                print(f"  {ant:<23} → {con:<18} {rule['confidence']:>7.2%} {rule['lift']:>7.2f}")
        
        print("\n" + "="*70)
    
    def export_rules_json(self, filepath: str = "rules.json"):
        """Export rules to JSON for API use."""
        if self.rules is None:
            return
        
        rules_list = []
        for _, rule in self.rules.iterrows():
            rules_list.append({
                'antecedents': list(rule['antecedents']),
                'consequents': list(rule['consequents']),
                'support': round(rule['support'], 4),
                'confidence': round(rule['confidence'], 4),
                'lift': round(rule['lift'], 4)
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_list, f, indent=2)
        
        print(f"Exported {len(rules_list)} rules to {filepath}")
    
    def save_model(self, filepath: str = "mba_model.json"):
        """Save the model state for later use."""
        model_data = {
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'item_frequencies': dict(self.item_frequencies),
            'transactions_count': len(self.transactions),
            'rules': []
        }
        
        if self.rules is not None:
            for _, rule in self.rules.iterrows():
                model_data['rules'].append({
                    'antecedents': list(rule['antecedents']),
                    'consequents': list(rule['consequents']),
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift'])
                })
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")


# Example usage and demonstration
if __name__ == "__main__":
    # Import the dataset generator
    from dataset_generator import generate_dataset, get_transactions_as_lists
    
    # Generate dataset
    print("Generating dataset...")
    raw_transactions = generate_dataset(200)
    transactions = get_transactions_as_lists(raw_transactions)
    
    # Initialize analyzer
    analyzer = MarketBasketAnalyzer(min_support=0.03, min_confidence=0.15)
    
    # Load and analyze
    analyzer.load_transactions(transactions)
    analyzer.run_apriori()
    analyzer.generate_rules(metric="lift", min_threshold=1.0)
    
    # Print comprehensive report
    analyzer.print_analysis_report()
    
    # Test recommendations
    print("\n🛒 RECOMMENDATION TEST")
    print("-" * 40)
    test_carts = [
        ["Milk", "Bread"],
        ["Rice", "Dal"],
        ["Chips"],
        ["Soap", "Shampoo"]
    ]
    
    for cart in test_carts:
        print(f"\nCart: {cart}")
        recommendations = analyzer.get_recommendations(cart, top_n=3)
        if recommendations:
            for rec in recommendations:
                print(f"  → {rec['item']} (confidence: {rec['confidence']:.0%})")
        else:
            print("  No recommendations found")
    
    # Export for API use
    analyzer.export_rules_json("rules.json")
    analyzer.save_model("mba_model.json")
