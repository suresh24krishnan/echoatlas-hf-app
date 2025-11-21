from utils.region_loader import get_region_info

def get_customs(location):
    """
    Returns gesture and etiquette tips for a given location.
    Falls back to 'Unknown' if data is missing.
    """
    region_data = get_region_info(location)
    return {
        "gesture": region_data.get("gesture", "Unknown"),
        "custom": region_data.get("custom", "No data available")
    }
