def detect_location(mode, user_input_location):
    """
    Returns the resolved location based on user input and selected mode.
    Defaults to Tokyo for International, Chennai for Indian States.
    """
    if user_input_location:
        return user_input_location
    return "Tokyo, Japan" if mode == "🌐 International" else "Chennai, Tamil Nadu"
