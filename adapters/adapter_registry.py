import os
import inspect
import importlib
from adapters.base_adapter import BaseAdapter

def find_adapter_classes():
    '''
    helper to return adapters and class methods from the directory of this file
    '''
    directory = os.path.dirname(__file__)
    adapter_classes = []
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            # Construct the absolute module path
            absolute_module_path = f"adapters.{module_name}"
            module = importlib.import_module(absolute_module_path)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                    class_methods = [method[0] for method in inspect.getmembers(obj, inspect.isfunction)]
                    adapter_classes.append((name, class_methods))
    return adapter_classes
