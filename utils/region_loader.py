import json

def get_region_info(region_or_location: str):
    """
    Look up region information by either its clean key (e.g., "United States (New York)")
    or by its location string (e.g., "New York, USA").
    
    Returns the full metadata dictionary for the region:
    {
        "location": "...",
        "emoji": "...",
        "description": "...",
        "phrase": "...",
        "gesture": "...",
        "custom": "...",
        "tone": "..."
    }
    """
    try:
        with open("regions.json", "r", encoding="utf-8") as f:
            all_regions = json.load(f)

        # Flatten all regions into one dict
        flat_regions = {
            region: info
            for category in all_regions.values()
            for region, info in category.items()
        }

        # Try direct region key match
        if region_or_location in flat_regions:
            return flat_regions[region_or_location]

        # Try location match
        for info in flat_regions.values():
            if info.get("location") == region_or_location:
                return info

    except Exception as e:
        print(f"Region loader failed: {e}")

    return None
