from utils.region_loader import get_region_info

def adjust_tone(location):
    """
    Returns the culturally appropriate emotional tone for a given location.
    Falls back to 'unknown' if tone is not defined.
    """
    return get_region_info(location).get("tone", "unknown")
