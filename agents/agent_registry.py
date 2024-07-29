import os
import inspect
import importlib.util
from agents.base_agent import BaseAgent  # Adjust the import according to your project structure

def get_all_base_agent_classes():
    base_agent_classes = []
    directory = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
    
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            module_path = os.path.join(directory, filename)
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                    base_agent_classes.append(obj)
    
    return base_agent_classes