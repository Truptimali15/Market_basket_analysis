# dynamic_updater.py
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque
import threading
from market_basket_analysis import MarketBasketAnalyzer


class DynamicMBAUpdater:
    """
    Handle dynamic updates to the Market Basket Analysis model.
    
    Two approaches:
    1. BATCH UPDATE: Re-run full Apriori periodically (recommended for most cases)
    2. INCREMENTAL: Approximate updates without full recalculation
    
    For most retail scenarios, batch updates (e.g., nightly) are sufficient
    and provide more accurate results than incremental approaches.
    """
    
    def __init__(self, 
                 analyzer: MarketBasketAnalyzer,
                 update_threshold: int = 50,
                 auto_update: bool = True):
        """
        Initialize the dynamic updater.
        
        Args:
            analyzer: The MBA analyzer instance
            update_threshold: Number of new transactions before triggering update
            auto_update: Whether to automatically update when threshold reached
        """
        self.analyzer = analyzer
        self.update_threshold = update_threshold
        self.auto_update = auto_update
        
        # Buffer for new transactions
        self.transaction_buffer: deque = deque(maxlen=10000)
        self.new_transactions_count = 0
        self.last_update = datetime.now()
        
        # Statistics
        self.update_history: List[Dict] = []
        
    def add_transaction(self, items: List[str], metadata: Optional[Dict] = None) -> Dict:
        """
        Add a new transaction to the system.
        
        Args:
            items: List of items in the transaction
            metadata: Optional metadata (timestamp, customer_id, etc.)
            
        Returns:
            Status dict with recommendations based on current model
        """
        timestamp = datetime.now().isoformat()
        
        transaction = {
            'items': items,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Add to buffer
        self.transaction_buffer.append(transaction)
        self.new_transactions_count += 1
        
        # Get recommendations from current model
        recommendations = self.analyzer.get_recommendations(items, top_n=3)
        
        # Check if update needed
        update_triggered = False
        if self.auto_update and self.new_transactions_count >= self.update_threshold:
            self._trigger_batch_update()
            update_triggered = True
        
        return {
            'status': 'success',
            'transaction_recorded': True,
            'recommendations': recommendations,
            'pending_updates': self.new_transactions_count,
            'update_triggered': update_triggered
        }
    
    def _trigger_batch_update(self):
        """Trigger a batch update of the model."""
        print(f"\n🔄 Triggering batch update ({self.new_transactions_count} new transactions)...")
        
        start_time = time.time()
        
        # Add buffered transactions to main dataset
        new_items_list = [txn['items'] for txn in self.transaction_buffer]
        
        # Combine with existing transactions
        all_transactions = self.analyzer.transactions + new_items_list
        
        # Reload and rerun analysis
        self.analyzer.load_transactions(all_transactions)
        self.analyzer.run_apriori()
        self.analyzer.generate_rules()
        
        elapsed = time.time() - start_time
        
        # Record update
        self.update_history.append({
            'timestamp': datetime.now().isoformat(),
            'transactions_added': self.new_transactions_count,
            'total_transactions': len(all_transactions),
            'rules_generated': len(self.analyzer.rules) if self.analyzer.rules is not None else 0,
            'processing_time': round(elapsed, 2)
        })
        
        # Reset counter and buffer
        self.new_transactions_count = 0
        self.transaction_buffer.clear()
        self.last_update = datetime.now()
        
        print(f"✅ Update complete in {elapsed:.2f}s")
        print(f"   Total transactions: {len(all_transactions)}")
        print(f"   Rules generated: {len(self.analyzer.rules)}")
    
    def force_update(self):
        """Force an immediate update regardless of threshold."""
        if self.new_transactions_count > 0:
            self._trigger_batch_update()
        else:
            print("No new transactions to process")
    
    def get_update_status(self) -> Dict:
        """Get current update status."""
        return {
            'new_transactions_pending': self.new_transactions_count,
            'update_threshold': self.update_threshold,
            'last_update': self.last_update.isoformat(),
            'total_transactions': len(self.analyzer.transactions),
            'buffer_size': len(self.transaction_buffer),
            'update_history': self.update_history[-5:]  # Last 5 updates
        }
    
    def schedule_periodic_update(self, interval_seconds: int = 3600):
        """
        Schedule periodic updates (for production use).
        
        Args:
            interval_seconds: Update interval (default: 1 hour)
        """
        def update_job():
            while True:
                time.sleep(interval_seconds)
                if self.new_transactions_count > 0:
                    self._trigger_batch_update()
        
        thread = threading.Thread(target=update_job, daemon=True)
        thread.start()
        print(f"📅 Scheduled updates every {interval_seconds}s")
    
    def save_state(self, filepath: str = "mba_state.json"):
        """Save current state for persistence."""
        state = {
            'analyzer_state': {
                'transactions_count': len(self.analyzer.transactions),
                'min_support': self.analyzer.min_support,
                'min_confidence': self.analyzer.min_confidence
            },
            'buffer': list(self.transaction_buffer),
            'new_transactions_count': self.new_transactions_count,
            'last_update': self.last_update.isoformat(),
            'update_history': self.update_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"State saved to {filepath}")


class IncrementalApproximator:
    """
    Approximate incremental updates for real-time scenarios.
    
    NOTE: This provides faster but less accurate updates.
    Use batch updates for authoritative results.
    """
    
    def __init__(self, analyzer: MarketBasketAnalyzer):
        self.analyzer = analyzer
        self.item_pair_counts = {}
        self.total_transactions = 0
        self._initialize_counts()
    
    def _initialize_counts(self):
        """Initialize pair counts from existing transactions."""
        for txn in self.analyzer.transactions:
            self.total_transactions += 1
            items = sorted(set(txn))
            
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    pair = (item1, item2)
                    self.item_pair_counts[pair] = self.item_pair_counts.get(pair, 0) + 1
    
    def update_with_transaction(self, items: List[str]):
        """Update counts with a new transaction (incremental)."""
        self.total_transactions += 1
        items = sorted(set(items))
        
        # Update pair counts
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                pair = (item1, item2)
                self.item_pair_counts[pair] = self.item_pair_counts.get(pair, 0) + 1
    
    def get_approximate_support(self, itemset: tuple) -> float:
        """Get approximate support for a pair."""
        if len(itemset) == 2:
            pair = tuple(sorted(itemset))
            count = self.item_pair_counts.get(pair, 0)
            return count / self.total_transactions if self.total_transactions > 0 else 0
        return 0
    
    def get_top_pairs(self, top_n: int = 10) -> List[tuple]:
        """Get top co-occurring pairs."""
        sorted_pairs = sorted(
            self.item_pair_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_pairs[:top_n]


# Demonstration
if __name__ == "__main__":
    from dataset_generator import generate_dataset, get_transactions_as_lists
    
    # Initial setup
    print("="*60)
    print("DYNAMIC UPDATE DEMONSTRATION")
    print("="*60)
    
    # Create initial dataset
    initial_txns = get_transactions_as_lists(generate_dataset(100))
    
    analyzer = MarketBasketAnalyzer(min_support=0.03, min_confidence=0.15)
    analyzer.load_transactions(initial_txns)
    analyzer.run_apriori()
    analyzer.generate_rules()
    
    print(f"\nInitial model: {len(analyzer.rules)} rules")
    
    # Create updater
    updater = DynamicMBAUpdater(analyzer, update_threshold=20)
    
    # Simulate new transactions
    print("\n--- Simulating incoming transactions ---")
    new_transactions = [
        ["Milk", "Bread", "Butter"],
        ["Chips", "Soft Drinks", "Namkeen"],
        ["Rice", "Dal", "Cooking Oil", "Onion"],
        ["Soap", "Shampoo", "Toothpaste"],
        ["Tea", "Sugar", "Biscuits"],
    ]
    
    for i, txn in enumerate(new_transactions):
        result = updater.add_transaction(txn)
        print(f"\nTransaction {i+1}: {txn}")
        print(f"  Recommendations: {[r['item'] for r in result['recommendations']]}")
        print(f"  Pending updates: {result['pending_updates']}")
    
    # Check status
    print("\n--- Update Status ---")
    status = updater.get_update_status()
    print(f"New transactions pending: {status['new_transactions_pending']}")
    print(f"Last update: {status['last_update']}")
    
    # Force update
    print("\n--- Forcing Update ---")
    # Add more to hit threshold
    for _ in range(20):
        updater.add_transaction(["Milk", "Tea", "Sugar"])
    
    print(f"\nFinal model: {len(analyzer.rules)} rules")
