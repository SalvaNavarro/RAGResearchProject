# misc utility functions

def add_zeroes(idx) -> str:
    """
    Utility file name formatting method.

    Returns:
        None
    """
    if idx < 10:
        formatted_idx = "00"
    elif idx < 100:
        formatted_idx = "0"
    else:
        formatted_idx = ""
    
    return formatted_idx