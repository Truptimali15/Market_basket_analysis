# dataset_generator.py
import random
import csv
from datetime import datetime, timedelta
from collections import defaultdict

# Define product categories and items with realistic purchase probabilities
PRODUCT_CATALOG = {
    "Dairy": {
        "Milk": 0.35,
        "Curd": 0.20,
        "Butter": 0.15,
        "Cheese": 0.12,
        "Paneer": 0.18,
        "Ghee": 0.08
    },
    "Bakery": {
        "Bread": 0.30,
        "Biscuits": 0.25,
        "Cake": 0.10,
        "Cookies": 0.15,
        "Rusk": 0.12
    },
    "Groceries": {
        "Rice": 0.25,
        "Wheat Flour": 0.20,
        "Sugar": 0.18,
        "Salt": 0.10,
        "Cooking Oil": 0.22,
        "Dal": 0.20,
        "Tea": 0.15,
        "Coffee": 0.12
    },
    "Snacks": {
        "Chips": 0.22,
        "Namkeen": 0.18,
        "Chocolates": 0.20,
        "Candy": 0.12,
        "Popcorn": 0.08
    },
    "Beverages": {
        "Soft Drinks": 0.18,
        "Juice": 0.15,
        "Mineral Water": 0.20,
        "Energy Drink": 0.08
    },
    "Personal Care": {
        "Soap": 0.20,
        "Shampoo": 0.15,
        "Toothpaste": 0.18,
        "Face Wash": 0.10,
        "Deodorant": 0.08
    },
    "Household": {
        "Detergent": 0.18,
        "Dishwash": 0.15,
        "Floor Cleaner": 0.10,
        "Toilet Cleaner": 0.08
    },
    "Fruits & Vegetables": {
        "Onion": 0.28,
        "Tomato": 0.25,
        "Potato": 0.30,
        "Banana": 0.22,
        "Apple": 0.15
    }
}

# Define realistic item associations (items frequently bought together)
ITEM_ASSOCIATIONS = {
    "Milk": ["Bread", "Sugar", "Tea", "Coffee", "Biscuits"],
    "Bread": ["Milk", "Butter", "Eggs", "Cheese"],
    "Tea": ["Sugar", "Milk", "Biscuits"],
    "Coffee": ["Sugar", "Milk", "Cookies"],
    "Rice": ["Dal", "Cooking Oil", "Onion", "Tomato"],
    "Chips": ["Soft Drinks", "Chocolates", "Namkeen"],
    "Soft Drinks": ["Chips", "Namkeen", "Popcorn"],
    "Soap": ["Shampoo", "Toothpaste"],
    "Detergent": ["Dishwash", "Floor Cleaner"],
    "Onion": ["Tomato", "Potato", "Cooking Oil"],
    "Banana": ["Apple", "Milk", "Curd"]
}

# Add Eggs to catalog (common item)
PRODUCT_CATALOG["Dairy"]["Eggs"] = 0.28


def generate_transaction(transaction_id: int, base_date: datetime) -> dict:
    """Generate a single realistic transaction."""
    
    # Random transaction date within a month
    transaction_date = base_date + timedelta(
        days=random.randint(0, 30),
        hours=random.randint(8, 21),
        minutes=random.randint(0, 59)
    )
    
    # Determine basket size (typically 3-12 items)
    basket_size = random.choices(
        range(2, 15),
        weights=[5, 15, 20, 18, 15, 10, 7, 4, 3, 2, 1, 0.5, 0.5],
        k=1
    )[0]
    
    items = []
    
    # Select initial items based on category probabilities
    all_items = []
    for category, products in PRODUCT_CATALOG.items():
        for product, prob in products.items():
            all_items.append((product, prob, category))
    
    # Sort by probability and pick items
    random.shuffle(all_items)
    
    for product, prob, category in all_items:
        if len(items) >= basket_size:
            break
        if random.random() < prob and product not in items:
            items.append(product)
            
            # Add associated items with some probability
            if product in ITEM_ASSOCIATIONS and len(items) < basket_size:
                for associated in ITEM_ASSOCIATIONS[product]:
                    if random.random() < 0.4 and associated not in items:
                        items.append(associated)
                        if len(items) >= basket_size:
                            break
    
    # Ensure at least 2 items
    if len(items) < 2:
        available = [p for p, _, _ in all_items if p not in items]
        items.extend(random.sample(available, min(2 - len(items), len(available))))
    
    return {
        "transaction_id": f"TXN{transaction_id:04d}",
        "date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
        "items": items
    }


def generate_dataset(num_transactions: int = 200) -> list:
    """Generate complete dataset with specified number of transactions."""
    
    base_date = datetime(2024, 1, 1)
    transactions = []
    
    for i in range(1, num_transactions + 1):
        txn = generate_transaction(i, base_date)
        transactions.append(txn)
    
    return transactions


def save_as_csv(transactions: list, filename: str = "transactions.csv"):
    """Save transactions to CSV file."""
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["transaction_id", "date", "items"])
        
        for txn in transactions:
            writer.writerow([
                txn["transaction_id"],
                txn["date"],
                ",".join(txn["items"])
            ])
    
    print(f"Saved {len(transactions)} transactions to {filename}")


def save_as_basket_format(transactions: list, filename: str = "baskets.csv"):
    """Save in basket format (one-hot encoded) for mlxtend."""
    
    # Get all unique items
    all_items = set()
    for txn in transactions:
        all_items.update(txn["items"])
    
    all_items = sorted(list(all_items))
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["transaction_id"] + all_items)
        
        for txn in transactions:
            row = [txn["transaction_id"]]
            for item in all_items:
                row.append(1 if item in txn["items"] else 0)
            writer.writerow(row)
    
    print(f"Saved basket format to {filename}")


def get_transactions_as_lists(transactions: list) -> list:
    """Return transactions as list of lists (for direct use in Python)."""
    return [txn["items"] for txn in transactions]


# Generate and display sample data
if __name__ == "__main__":
    # Generate dataset
    transactions = generate_dataset(200)
    
    # Save in different formats
    save_as_csv(transactions, "transactions.csv")
    save_as_basket_format(transactions, "baskets.csv")
    
    # Display sample
    print("\n--- Sample Transactions ---")
    for txn in transactions[:10]:
        print(f"{txn['transaction_id']} | {txn['date']} | {', '.join(txn['items'])}")
    
    # Statistics
    print("\n--- Dataset Statistics ---")
    total_items = sum(len(txn["items"]) for txn in transactions)
    print(f"Total transactions: {len(transactions)}")
    print(f"Total items sold: {total_items}")
    print(f"Average basket size: {total_items / len(transactions):.2f}")
    
    # Item frequency
    item_counts = defaultdict(int)
    for txn in transactions:
        for item in txn["items"]:
            item_counts[item] += 1
    
    print("\n--- Top 10 Most Frequent Items ---")
    for item, count in sorted(item_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {item}: {count} ({count/len(transactions)*100:.1f}%)")
