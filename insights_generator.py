# insights_generator.py
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from market_basket_analysis import MarketBasketAnalyzer


class ShopInsightsGenerator:
    """Generate actionable insights for shop owners."""
    
    def __init__(self, analyzer: MarketBasketAnalyzer):
        self.analyzer = analyzer
        
    def get_bundling_strategies(self, min_lift: float = 1.5, top_n: int = 10) -> List[Dict]:
        """
        Identify products that should be bundled together for promotions.
        
        High lift values indicate strong associations worth bundling.
        """
        bundles = []
        
        if self.analyzer.rules is None:
            return bundles
        
        # Filter for high-lift rules
        high_lift_rules = self.analyzer.rules[self.analyzer.rules['lift'] >= min_lift]
        
        seen_combinations = set()
        
        for _, rule in high_lift_rules.head(top_n * 2).iterrows():
            items = tuple(sorted(list(rule['antecedents']) + list(rule['consequents'])))
            
            if items not in seen_combinations and len(items) >= 2:
                seen_combinations.add(items)
                
                # Calculate potential bundle price
                base_items = list(items)
                
                bundles.append({
                    'bundle_name': f"{' + '.join(items[:2])} Bundle",
                    'items': list(items),
                    'lift': round(rule['lift'], 2),
                    'confidence': round(rule['confidence'], 2),
                    'recommendation': self._get_bundle_recommendation(rule['lift']),
                    'pricing_suggestion': "Offer 10-15% discount on bundle"
                })
                
                if len(bundles) >= top_n:
                    break
        
        return bundles
    
    def _get_bundle_recommendation(self, lift: float) -> str:
        if lift > 3:
            return "⭐ STRONG: Create permanent combo offer"
        elif lift > 2:
            return "✓ GOOD: Weekly featured bundle"
        else:
            return "○ MODERATE: Seasonal promotion"
    
    def get_placement_suggestions(self) -> List[Dict]:
        """
        Suggest which products should be placed together in the store.
        
        Based on frequently bought together items (high support pairs).
        """
        suggestions = []
        
        if self.analyzer.frequent_itemsets is None:
            return suggestions
        
        # Get pairs with high support
        pairs = self.analyzer.frequent_itemsets[
            self.analyzer.frequent_itemsets['length'] == 2
        ].nlargest(15, 'support')
        
        # Group by sections
        dairy_items = {'Milk', 'Curd', 'Butter', 'Cheese', 'Paneer', 'Eggs', 'Ghee'}
        grocery_items = {'Rice', 'Wheat Flour', 'Sugar', 'Salt', 'Cooking Oil', 'Dal', 'Tea', 'Coffee'}
        snack_items = {'Chips', 'Namkeen', 'Chocolates', 'Candy', 'Popcorn', 'Biscuits', 'Cookies'}
        
        for _, row in pairs.iterrows():
            items = list(row['itemsets'])
            
            # Determine if cross-category placement
            categories = []
            for item in items:
                if item in dairy_items:
                    categories.append('Dairy')
                elif item in grocery_items:
                    categories.append('Grocery')
                elif item in snack_items:
                    categories.append('Snacks')
                else:
                    categories.append('Other')
            
            is_cross_category = len(set(categories)) > 1
            
            suggestions.append({
                'items': items,
                'support': round(row['support'], 3),
                'placement': 'Adjacent shelves' if is_cross_category else 'Same section',
                'action': f"Place {items[0]} near {items[1]}",
                'reason': f"Bought together in {row['support']*100:.1f}% of relevant transactions"
            })
        
        return suggestions
    
    def get_cross_selling_ideas(self) -> List[Dict]:
        """
        Generate cross-selling ideas based on association rules.
        
        Identify trigger products that lead to other purchases.
        """
        ideas = []
        
        if self.analyzer.rules is None:
            return ideas
        
        # Find rules where antecedent is a single item (trigger product)
        single_trigger_rules = self.analyzer.rules[
            self.analyzer.rules['antecedents'].apply(len) == 1
        ]
        
        # Group by trigger item
        trigger_suggestions = defaultdict(list)
        
        for _, rule in single_trigger_rules.iterrows():
            trigger = list(rule['antecedents'])[0]
            consequent = list(rule['consequents'])
            
            trigger_suggestions[trigger].append({
                'suggest_items': consequent,
                'confidence': rule['confidence'],
                'lift': rule['lift']
            })
        
        # Create actionable ideas
        for trigger, suggestions in trigger_suggestions.items():
            # Sort by confidence
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            top_suggestion = suggestions[0]
            
            ideas.append({
                'trigger_product': trigger,
                'cross_sell_items': top_suggestion['suggest_items'],
                'confidence': round(top_suggestion['confidence'], 2),
                'implementation': self._get_cross_sell_implementation(trigger, top_suggestion)
            })
        
        # Sort by confidence
        ideas.sort(key=lambda x: x['confidence'], reverse=True)
        return ideas[:15]
    
    def _get_cross_sell_implementation(self, trigger: str, suggestion: Dict) -> str:
        items = ', '.join(suggestion['suggest_items'])
        
        if suggestion['confidence'] > 0.5:
            return f"At checkout: 'Add {items} to your cart?' (popup suggestion)"
        elif suggestion['confidence'] > 0.3:
            return f"Display '{items}' near {trigger} section"
        else:
            return f"Email campaign: Bundle offers with {trigger}"
    
    def get_time_based_insights(self, transactions_with_time: List[Dict]) -> Dict:
        """Analyze purchasing patterns by time (if timestamp available)."""
        # This would analyze peak hours, day-of-week patterns, etc.
        # Placeholder for time-based analysis
        return {
            'peak_hours': '10 AM - 1 PM, 5 PM - 8 PM',
            'busiest_day': 'Saturday',
            'recommendation': 'Ensure high-demand items are stocked before peak hours'
        }
    
    def generate_full_report(self) -> Dict:
        """Generate a comprehensive insights report."""
        return {
            'bundling_strategies': self.get_bundling_strategies(),
            'placement_suggestions': self.get_placement_suggestions(),
            'cross_selling_ideas': self.get_cross_selling_ideas(),
            'summary': {
                'total_products': len(self.analyzer.item_frequencies),
                'total_transactions': len(self.analyzer.transactions),
                'rules_generated': len(self.analyzer.rules) if self.analyzer.rules is not None else 0
            }
        }
    
    def print_insights_report(self):
        """Print formatted insights for shop owner."""
        print("\n" + "="*70)
        print("       💡 ACTIONABLE INSIGHTS FOR SUDHIR SUPERSHOPY")
        print("="*70)
        
        # Bundling strategies
        print("\n📦 PRODUCT BUNDLING OPPORTUNITIES")
        print("-" * 50)
        bundles = self.get_bundling_strategies(top_n=5)
        for i, bundle in enumerate(bundles, 1):
            print(f"\n{i}. {bundle['bundle_name']}")
            print(f"   Items: {', '.join(bundle['items'])}")
            print(f"   Lift: {bundle['lift']} | {bundle['recommendation']}")
            print(f"   💰 {bundle['pricing_suggestion']}")
        
        # Placement suggestions
        print("\n\n🏪 STORE LAYOUT RECOMMENDATIONS")
        print("-" * 50)
        placements = self.get_placement_suggestions()[:8]
        for p in placements:
            print(f"\n  ➤ {p['action']}")
            print(f"    Reason: {p['reason']}")
            print(f"    Placement: {p['placement']}")
        
        # Cross-selling
        print("\n\n🛒 CROSS-SELLING STRATEGIES")
        print("-" * 50)
        cross_sells = self.get_cross_selling_ideas()[:8]
        for cs in cross_sells:
            print(f"\n  When customer buys: {cs['trigger_product']}")
            print(f"  Suggest: {', '.join(cs['cross_sell_items'])}")
            print(f"  Confidence: {cs['confidence']:.0%}")
            print(f"  How: {cs['implementation']}")
        
        print("\n" + "="*70)


# Run insights generation
if __name__ == "__main__":
    from dataset_generator import generate_dataset, get_transactions_as_lists
    
    # Generate and analyze
    transactions = get_transactions_as_lists(generate_dataset(200))
    
    analyzer = MarketBasketAnalyzer(min_support=0.03, min_confidence=0.15)
    analyzer.load_transactions(transactions)
    analyzer.run_apriori()
    analyzer.generate_rules()
    
    # Generate insights
    insights = ShopInsightsGenerator(analyzer)
    insights.print_insights_report()
