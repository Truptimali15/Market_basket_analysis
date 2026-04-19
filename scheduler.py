# scheduler.py
import schedule
import time
import logging
from datetime import datetime, timedelta
from threading import Thread

from database_models import MongoDBConfig, TransactionRepository, RulesRepository, AssociationRuleDocument
from market_basket_analysis import MarketBasketAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MBAScheduler:
    """
    Scheduler for running Market Basket Analysis jobs.
    
    Recommendation: Run batch updates based on your transaction volume:
    - High volume (1000+ transactions/day): Every 4-6 hours
    - Medium volume (100-1000/day): Daily (overnight)
    - Low volume (<100/day): Weekly
    """
    
    def __init__(self, db_config: MongoDBConfig):
        self.db = db_config
        self.txn_repo = TransactionRepository(db_config)
        self.rules_repo = RulesRepository(db_config)
        self.analyzer = MarketBasketAnalyzer(min_support=0.02, min_confidence=0.1)
        self.last_run = None
        
    def run_analysis(self):
        """Run full Market Basket Analysis and update rules."""
        logger.info(f"Starting MBA analysis at {datetime.now()}")
        start_time = time.time()
        
        try:
            # Get all transactions
            transactions = self.txn_repo.get_all_items_lists()
            
            if len(transactions) < 10:
                logger.warning("Not enough transactions for meaningful analysis")
                return
            
            # Run analysis
            self.analyzer.load_transactions(transactions)
            self.analyzer.run_apriori()
            self.analyzer.generate_rules(metric="lift", min_threshold=1.0)
            
            # Store rules in database
            rules_to_store = []
            if self.analyzer.rules is not None:
                for _, rule in self.analyzer.rules.iterrows():
                    rules_to_store.append(AssociationRuleDocument(
                        antecedents=list(rule['antecedents']),
                        consequents=list(rule['consequents']),
                        support=float(rule['support']),
                        confidence=float(rule['confidence']),
                        lift=float(rule['lift']),
                        model_version="1.0"
                    ))
            
            self.rules_repo.bulk_insert(rules_to_store)
            
            elapsed = time.time() - start_time
            self.last_run = datetime.now()
            
            logger.info(f"MBA analysis complete: {len(rules_to_store)} rules generated in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in MBA analysis: {e}")
    
    def run_incremental_update(self):
        """Run incremental update for recent transactions only."""
        if self.last_run is None:
            self.run_analysis()
            return
        
        # Get transactions since last run
        recent = self.txn_repo.get_by_date_range(self.last_run, datetime.now())
        
        if len(recent) < 10:
            logger.info("Not enough new transactions for incremental update")
            return
        
        logger.info(f"Running incremental update with {len(recent)} new transactions")
        # For now, just run full analysis
        # True incremental algorithms (FP-Growth streaming) could be implemented here
        self.run_analysis()
    
    def start_scheduler(self, daily_time: str = "02:00"):
        """
        Start the scheduler for daily updates.
        
        Args:
            daily_time: Time to run daily analysis (24h format)
        """
        # Schedule daily full analysis
        schedule.every().day.at(daily_time).do(self.run_analysis)
        
        # Schedule hourly incremental checks (for high-volume stores)
        # schedule.every().hour.do(self.run_incremental_update)
        
        logger.info(f"Scheduler started. Daily analysis at {daily_time}")
        
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        thread = Thread(target=run_schedule, daemon=True)
        thread.start()
        
        return thread


# Example usage
if __name__ == "__main__":
    db = MongoDBConfig()
    scheduler = MBAScheduler(db)
    
    # Run initial analysis
    scheduler.run_analysis()
    
    # Start scheduled jobs
    scheduler.start_scheduler(daily_time="02:00")
    
    # Keep main thread alive
    while True:
        time.sleep(3600)
