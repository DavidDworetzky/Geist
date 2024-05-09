import os
import inspect
import importlib
from typing import List
from adapters.base_adapter import BaseAdapter

def _get_adapter_files() -> List[str]:
    directory = os.path.dirname(__file__)
    adapter_classes = [filename for filename in os.listdir(directory) if filename.endswith('.py') and not filename.startswith('__')]
    return adapter_classes

def _get_class_module(module_name: str) -> str:
    return f"adapters.{module_name}"

def find_adapter_classes():
    '''
    helper to return adapters and class methods from the directory of this file
    '''
    adapter_classes = []
    filenames = _get_adapter_files()
    for filename in filenames:
        module_name = filename[:-3]  # Remove .py extension
        # Construct the absolute module path
        absolute_module_path = _get_class_module(module_name)
        module = importlib.import_module(absolute_module_path)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                class_methods = [method[0] for method in inspect.getmembers(obj, inspect.isfunction)]
                adapter_classes.append((name, class_methods))
    return adapter_classes

def init_adapter_class(classname: str, **kwargs):
    '''
    Dynamically initializes an adapter class by name with the provided kwargs.
    '''
    filenames = _get_adapter_files()
    for filename in filenames:
        module_name = filename[:-3]  # Remove .py extension
        absolute_module_path = _get_class_module(module_name)
        module = importlib.import_module(absolute_module_path)
        # Check if the class exists in the module
        if hasattr(module, classname):
            adapter_class = getattr(module, classname)
            if issubclass(adapter_class, BaseAdapter):
                # Instantiate the class with kwargs
                return adapter_class(**kwargs)
    raise ValueError(f"Adapter class {classname} not found in adapters directory.")
