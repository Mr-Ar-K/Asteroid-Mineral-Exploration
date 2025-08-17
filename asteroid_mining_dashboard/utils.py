# utils.py

import numpy as np

def calculate_final_aps(row):
    """
    Calculates the final Asteroid Potential Score (APS) based on a weighted system.
    """
    # Part 1: Resource Value (out of 50 points)
    resource_value = 0
    if row['predicted_class'] == 'M':
        resource_value = 50  # Highest value for metals
    elif row['predicted_class'] == 'C':
        resource_value = 35  # High value for water
    elif row['predicted_class'] == 'S':
        resource_value = 15  # Lower value for stony materials

    # Part 2: Mission Accessibility (out of 50 points)
    mission_value = (row.get('mission_accessibility_score', 0)) * 50

    # Part 3: Combine and apply the confidence penalty
    raw_aps = resource_value + mission_value
    confidence_score = row.get('resource_confidence_score', 0.1)  # Default to low confidence
    final_aps = int(raw_aps * confidence_score)

    return final_aps


def get_resource_summary(predicted_class):
    """
    Provides a simple, user-friendly summary of the asteroid's resources.
    """
    if predicted_class == 'M':
        return "High-Value Metals (Iron, Nickel, Platinum)"
    elif predicted_class == 'C':
        return "Water-Rich & Carbonaceous"
    elif predicted_class == 'S':
        return "Stony Materials & Silicates"
    else:
        return "Unknown Composition"
