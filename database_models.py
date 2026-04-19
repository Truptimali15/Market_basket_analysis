# database_models.py
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId


class MongoDBConfig:
    """MongoDB configuration and connection."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017"):
        self.client = MongoClient(connection_string)
        self.db = self.client.sudhir_supershopy
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Create indexes for better query performance."""
        # Transactions collection
        self.db.transactions.create_index([("timestamp", DESCENDING)])
        self.db.transactions.create_index([("customer_id", ASCENDING)])
        
        # Products collection
        self.db.products.create_index([("name", ASCENDING)], unique=True)
        self.db.products.create_index([("category", ASCENDING)])
        
        # Rules collection
        self.db.association_rules.create_index([("lift", DESCENDING)])
        self.db.association_rules.create_index([("antecedents", ASCENDING)])


# Document models
class TransactionDocument(BaseModel):
    """Transaction document model."""
    transaction_id: str
    customer_id: Optional[str] = None
    items: List[str]
    total_amount: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {ObjectId: str}


class ProductDocument(BaseModel):
    """Product document model."""
    name: str
    category: str
    price: float
    description: Optional[str] = None
    image_url: Optional[str] = None
    in_stock: bool = True
    stock_quantity: int = 100
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AssociationRuleDocument(BaseModel):
    """Association rule document model."""
    antecedents: List[str]
    consequents: List[str]
    support: float
    confidence: float
    lift: float
    created_at: datetime = Field(default_factory=datetime.now)
    model_version: str = "1.0"


# Database operations
class TransactionRepository:
    """Repository for transaction operations."""
    
    def __init__(self, db_config: MongoDBConfig):
        self.collection = db_config.db.transactions
    
    def insert(self, transaction: TransactionDocument) -> str:
        result = self.collection.insert_one(transaction.dict())
        return str(result.inserted_id)
    
    def get_recent(self, limit: int = 1000) -> List[dict]:
        return list(self.collection.find().sort("timestamp", DESCENDING).limit(limit))
    
    def get_all_items_lists(self) -> List[List[str]]:
        """Get all transactions as list of item lists (for MBA)."""
        transactions = self.collection.find({}, {"items": 1})
        return [txn["items"] for txn in transactions]
    
    def get_by_date_range(self, start: datetime, end: datetime) -> List[dict]:
        return list(self.collection.find({
            "timestamp": {"$gte": start, "$lte": end}
        }))
    
    def count(self) -> int:
        return self.collection.count_documents({})


class RulesRepository:
    """Repository for association rules."""
    
    def __init__(self, db_config: MongoDBConfig):
        self.collection = db_config.db.association_rules
    
    def bulk_insert(self, rules: List[AssociationRuleDocument]):
        """Replace all rules with new ones."""
        self.collection.delete_many({})  # Clear old rules
        if rules:
            self.collection.insert_many([r.dict() for r in rules])
    
    def get_rules_for_items(self, items: List[str], limit: int = 10) -> List[dict]:
        """Get rules where antecedents match given items."""
        return list(self.collection.find({
            "antecedents": {"$all": items}
        }).sort("confidence", DESCENDING).limit(limit))
    
    def get_top_rules(self, limit: int = 50) -> List[dict]:
        return list(self.collection.find().sort("lift", DESCENDING).limit(limit))
