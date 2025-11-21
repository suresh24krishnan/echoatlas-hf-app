from utils.region_loader import get_region_info

def suggest_phrase(location, user_input):
    """
    Returns a culturally appropriate phrase for the given location.
    Ignores user_input for now, but can be used for future personalization.
    """
    return get_region_info(location).get("phrase", "Location not supported")
