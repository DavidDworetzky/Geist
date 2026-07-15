import importlib.util
import inspect
import os
import sys

from agents.base_agent import BaseAgent  # Adjust the import according to your project structure


def get_all_base_agent_classes():
    base_agent_classes = []
    directory = os.path.dirname(os.path.abspath(__file__))  # Get the current directory

    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"_geist_agent_scan_{filename[:-3]}"
            module_path = os.path.join(directory, filename)

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)

            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                    base_agent_classes.append(obj)

    return base_agent_classes
