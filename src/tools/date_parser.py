from datetime import datetime


def parse_datetime(date_string: str) -> datetime:
    """
    Converts a date/time string to a datetime object.

    Supports multiple formats (add more as needed).
    Handles potential errors gracefully.

    Args:
        date_string: The string representing the date and time.

    Returns:
        A datetime object if parsing is successful,
        or an error message if parsing fails.
    """
    supported_formats = [
        "%Y-%m-%d",  # e.g., 2023-10-27
        "%Y-%m-%dT%H:%M:%S",  # e.g., 2023-10-27T14:30:00
        "%m/%d/%Y",  # e.g., 10/27/2023
        "%m/%d/%Y %H:%M",  # e.g., 10/27/2023 15:45
        "%Y-%m-%d %H:%M:%S",
        # Add more formats as you encounter them
    ]

    for fmt in supported_formats:
        try:
            datetime_object = datetime.strptime(date_string, fmt)
            return datetime_object  # Return immediately if successful
        except ValueError:
            pass  # Try the next format

    return "Error: Could not parse date/time string. Format not recognized."
