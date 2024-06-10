import os
import inspect
import importlib
from typing import List, Any
from adapters.base_adapter import BaseAdapter
import inspect
from dataclasses import dataclass

@dataclass
class AdapterWrapper:
    name: str
    instance: Any

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


def init_adapter_class(classname: str, args: dict) -> AdapterWrapper:
    '''
    Dynamically initializes an adapter class by name with the provided kwargs,
    only using the kwargs that are valid for the class's constructor.
    '''
    filenames = _get_adapter_files()
    for filename in filenames:
        module_name = filename[:-3]  # Remove .py extension
        absolute_module_path = _get_class_module(module_name)
        module = importlib.import_module(absolute_module_path)
        if hasattr(module, classname):
            adapter_class = getattr(module, classname)
            if issubclass(adapter_class, BaseAdapter):
                # Get the signature of the constructor
                constructor_signature = inspect.signature(adapter_class.__init__)
                # Filter args to only include valid parameters, excluding 'self'
                valid_args = {k: v for k, v in args.items() if k in constructor_signature.parameters and k != 'self'}
                # Instantiate the class with filtered kwargs
                instance = adapter_class(**valid_args)
                wrapper = AdapterWrapper(name = classname, instance = instance)
                return wrapper
    raise ValueError(f"Adapter class {classname} not found in adapters directory.")