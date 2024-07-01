import logging

def log_function_call(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Calling function: {func}")
        return func(*args, **kwargs)
    return wrapper
