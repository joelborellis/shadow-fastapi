import re

def clean_text(input_text):
    """
    Cleans the input text by removing or replacing special characters to make it JSON-safe.

    :param input_text: The raw input text to clean.
    :return: A cleaned version of the text.
    """
    try:
        # Replace problematic characters
        # Replace unusual unicode characters with a placeholder (like empty space or appropriate character)
        cleaned_text = input_text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
        cleaned_text = re.sub(r'[\[\]{}]', '', cleaned_text)  # Remove brackets
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple whitespace with a single space
        return cleaned_text.strip()
    except Exception as e:
        raise ValueError(f"Error in cleaning text: {e}")