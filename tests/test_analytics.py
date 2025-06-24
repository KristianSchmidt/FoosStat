"""Test the analytics engine with sample data"""

from analytics import AnalyticsEngine

# Test with a sample game sequence
test_sequence = [
    'r5', 'r3', 'g_r',  # Red scores from 3-bar
    'b5', 'b5', 'b3', 'g_b',  # Blue scores from 3-bar with midfield recatch
    'r5', 'r3', 'b2', 'r2', 'g_r',  # Red steal, then 2-bar goal
    'b5', 'b3', 'r2', 'b2', 'b5', 'b3', 'g_b',  # Complex rally ending in blue goal
    'r5', 'b5', 'r5', 'r3', 'b2', 'g_b'  # Blue steal then goal
]

print("Testing analytics with sample sequence:")
print("Sequence:", ', '.join(test_sequence))
print()

stats = AnalyticsEngine.generate_full_stats(test_sequence)

print("RED TEAM STATS:")
for stat_name, stat_value in stats["red"].items():
    print(f"  {stat_name}: {stat_value}")

print("\nBLUE TEAM STATS:")
for stat_name, stat_value in stats["blue"].items():
    print(f"  {stat_name}: {stat_value}")
