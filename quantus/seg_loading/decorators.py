def extensions(*exts: list) -> dict:
    """
    A decorator to specify the acceptable file extensions for a function.

    Args:
        exts (list): List of acceptable file extensions.

    Returns:
        function: The decorated function with the specified extensions.
    """
    def decorator(func):
        if type(func) is not dict:
            out_dict = {}
            out_dict['func'] = func
            out_dict['exts'] = exts
            return out_dict
        func['exts'] = exts
        return func
    return decorator
