# api_server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from datetime import datetime
import uvicorn

# Import FileResponse for serving HTML
from fastapi.responses import FileResponse

# Import our modules
from market_basket_analysis import MarketBasketAnalyzer
from insights_generator import ShopInsightsGenerator
from dynamic_updater import DynamicMBAUpdater
from dataset_generator import generate_dataset, get_transactions_as_lists, PRODUCT_CATALOG


# Pydantic models for API
class CartRequest(BaseModel):
    items: List[str]

class TransactionRequest(BaseModel):
    items: List[str]
    customer_id: Optional[str] = None
    timestamp: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    cart_items: List[str]
    generated_at: str

class ProductInfo(BaseModel):
    name: str
    category: str
    price: float
    image_url: str
    in_stock: bool = True


# Initialize FastAPI app
app = FastAPI(
    title="Sudhir SuperShopy MBA API",
    description="Market Basket Analysis API for product recommendations",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
analyzer: MarketBasketAnalyzer = None
updater: DynamicMBAUpdater = None
insights_gen: ShopInsightsGenerator = None

# Product catalog with prices and images
PRODUCTS_DB = {}


def initialize_products():
    """Initialize product database with prices and images."""
    global PRODUCTS_DB
    
    # Base prices by category
    category_prices = {
        "Dairy": (30, 150),
        "Bakery": (20, 100),
        "Groceries": (40, 200),
        "Snacks": (20, 80),
        "Beverages": (20, 100),
        "Personal Care": (50, 300),
        "Household": (80, 400),
        "Fruits & Vegetables": (30, 100)
    }
    
    product_id = 1
    for category, items in PRODUCT_CATALOG.items():
        price_range = category_prices.get(category, (50, 200))
        
        for item_name in items.keys():
            import random
            price = round(random.uniform(price_range[0], price_range[1]), 2)
            
            PRODUCTS_DB[item_name] = {
                "id": product_id,
                "name": item_name,
                "category": category,
                "price": price,
                "image_url": f"/images/{item_name.lower().replace(' ', '_')}.jpg",
                "in_stock": random.random() > 0.1  # 90% in stock
            }
            product_id += 1


def initialize_model():
    """Initialize the MBA model with sample data."""
    global analyzer, updater, insights_gen
    
    print("🚀 Initializing Market Basket Analysis model...")
    
    # Generate initial dataset
    transactions = get_transactions_as_lists(generate_dataset(200))
    
    # Initialize analyzer
    analyzer = MarketBasketAnalyzer(min_support=0.03, min_confidence=0.1)
    analyzer.load_transactions(transactions)
    analyzer.run_apriori()
    analyzer.generate_rules(metric="lift", min_threshold=1.0)
    
    # Initialize updater and insights generator
    updater = DynamicMBAUpdater(analyzer, update_threshold=50)
    insights_gen = ShopInsightsGenerator(analyzer)
    
    print(f"✅ Model initialized with {len(transactions)} transactions")
    print(f"   Generated {len(analyzer.rules)} association rules")


# Startup event
@app.on_event("startup")
async def startup_event():
    initialize_products()
    initialize_model()


# API Endpoints

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("main.html")


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(cart: CartRequest):
    """
    Get product recommendations based on cart items.
    
    This is the main endpoint for the recommendation system.
    """
    if not cart.items:
        return RecommendationResponse(
            recommendations=[],
            cart_items=[],
            generated_at=datetime.now().isoformat()
        )
    
    # Get recommendations from analyzer
    recommendations = analyzer.get_recommendations(cart.items, top_n=5)
    
    # Enrich with product info
    enriched_recommendations = []
    for rec in recommendations:
        item_name = rec['item']
        if item_name in PRODUCTS_DB:
            product = PRODUCTS_DB[item_name]
            enriched_recommendations.append({
                **rec,
                'price': product['price'],
                'category': product['category'],
                'image_url': product['image_url'],
                'in_stock': product['in_stock']
            })
        else:
            enriched_recommendations.append(rec)
    
    return RecommendationResponse(
        recommendations=enriched_recommendations,
        cart_items=cart.items,
        generated_at=datetime.now().isoformat()
    )


@app.get("/api/products")
async def get_all_products():
    """Get all products in the catalog."""
    return {
        "products": list(PRODUCTS_DB.values()),
        "categories": list(PRODUCT_CATALOG.keys()),
        "total": len(PRODUCTS_DB)
    }


@app.get("/api/products/category/{category}")
async def get_products_by_category(category: str):
    """Get products filtered by category."""
    products = [p for p in PRODUCTS_DB.values() if p['category'].lower() == category.lower()]
    
    if not products:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    return {"products": products, "category": category}


@app.get("/api/products/{product_name}")
async def get_product(product_name: str):
    """Get details for a specific product."""
    if product_name not in PRODUCTS_DB:
        raise HTTPException(status_code=404, detail=f"Product '{product_name}' not found")
    
    product = PRODUCTS_DB[product_name]
    
    # Get related recommendations
    recommendations = analyzer.get_recommendations([product_name], top_n=4)
    
    return {
        "product": product,
        "frequently_bought_together": recommendations
    }


@app.post("/api/transaction")
async def record_transaction(transaction: TransactionRequest, background_tasks: BackgroundTasks):
    """
    Record a new transaction and get recommendations.
    
    This updates the model in the background when threshold is reached.
    """
    result = updater.add_transaction(
        items=transaction.items,
        metadata={
            "customer_id": transaction.customer_id,
            "timestamp": transaction.timestamp or datetime.now().isoformat()
        }
    )
    
    return {
        "status": "recorded",
        "transaction_items": transaction.items,
        "recommendations": result['recommendations'],
        "model_update_pending": result['pending_updates']
    }


@app.get("/api/analytics/top-items")
async def get_top_items(limit: int = 10):
    """Get top selling items."""
    item_stats = analyzer.get_item_statistics()
    
    return {
        "top_items": item_stats.head(limit).to_dict('records'),
        "total_unique_items": len(item_stats)
    }


@app.get("/api/analytics/top-combinations")
async def get_top_combinations(limit: int = 10):
    """Get most frequent item combinations."""
    top_itemsets = analyzer.get_top_itemsets(min_length=2, top_n=limit)
    
    combinations = []
    for _, row in top_itemsets.iterrows():
        combinations.append({
            "items": list(row['itemsets']),
            "support": round(row['support'], 4),
            "frequency_percent": f"{row['support'] * 100:.1f}%"
        })
    
    return {"top_combinations": combinations}


@app.get("/api/analytics/rules")
async def get_association_rules(limit: int = 20, min_lift: float = 1.0):
    """Get association rules with filtering."""
    if analyzer.rules is None:
        return {"rules": [], "message": "No rules generated"}
    
    filtered_rules = analyzer.rules[analyzer.rules['lift'] >= min_lift].head(limit)
    
    rules_list = []
    for _, rule in filtered_rules.iterrows():
        rules_list.append({
            "if_buy": list(rule['antecedents']),
            "then_buy": list(rule['consequents']),
            "confidence": round(rule['confidence'], 3),
            "lift": round(rule['lift'], 3),
            "support": round(rule['support'], 4)
        })
    
    return {
        "rules": rules_list,
        "total_rules": len(analyzer.rules),
        "filtered_count": len(rules_list)
    }


@app.get("/api/insights/bundling")
async def get_bundling_suggestions():
    """Get product bundling recommendations for shop owner."""
    bundles = insights_gen.get_bundling_strategies(top_n=10)
    return {"bundling_opportunities": bundles}


@app.get("/api/insights/placement")
async def get_placement_suggestions():
    """Get store layout and placement recommendations."""
    placements = insights_gen.get_placement_suggestions()
    return {"placement_suggestions": placements}


@app.get("/api/insights/cross-sell")
async def get_cross_sell_ideas():
    """Get cross-selling strategies."""
    ideas = insights_gen.get_cross_selling_ideas()
    return {"cross_selling_ideas": ideas}


@app.get("/api/model/status")
async def get_model_status():
    """Get current model status and statistics."""
    status = updater.get_update_status()
    
    return {
        "model_status": "active",
        "total_transactions": len(analyzer.transactions),
        "total_rules": len(analyzer.rules) if analyzer.rules is not None else 0,
        "frequent_itemsets": len(analyzer.frequent_itemsets) if analyzer.frequent_itemsets is not None else 0,
        "update_status": status,
        "parameters": {
            "min_support": analyzer.min_support,
            "min_confidence": analyzer.min_confidence
        }
    }


@app.post("/api/model/refresh")
async def refresh_model(background_tasks: BackgroundTasks):
    """Force a model refresh."""
    background_tasks.add_task(updater.force_update)
    return {"status": "refresh_scheduled", "message": "Model refresh initiated in background"}


# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
