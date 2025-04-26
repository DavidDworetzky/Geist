#!/usr/bin/env python3
"""
Script to generate docker-compose.yml from a JSON configuration file.

This script reads container definitions from a JSON file (e.g., containers.json)
and generates a corresponding docker-compose.yml file.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml  # Requires PyYAML installation: pip install PyYAML
reserved_template_names = ["backend"]

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define default paths relative to the script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "containers.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docker-compose.yml"

def load_container_config(config_path: Path) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Loads container configurations from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Tuple containing:
            - List of container configurations (dict) or None if error.
            - Message describing the result (str).
    """
    if not config_path.is_file():
        msg = f"Configuration file not found: {config_path}"
        logger.error(msg)
        return None, msg
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            msg = f"Invalid format in {config_path}: Expected a list of container definitions."
            logger.error(msg)
            return None, msg
        logger.info(f"Successfully loaded configuration from {config_path}")
        return data, f"Successfully loaded configuration from {config_path}"
    except json.JSONDecodeError as e:
        msg = f"Error decoding JSON from {config_path}: {e}"
        logger.error(msg)
        return None, msg
    except Exception as e:
        msg = f"An unexpected error occurred while reading {config_path}: {e}"
        logger.error(msg, exc_info=True)
        return None, msg

def generate_compose_dict(container_configs: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Generates the docker-compose dictionary structure from container configurations.

    Args:
        container_configs: A list of dictionaries, each representing a service configuration
                           as defined in the input JSON.

    Returns:
        Tuple containing:
            - The generated docker-compose dictionary or None if error.
            - Message describing the result (str).
    """
    compose_data: Dict[str, Any] = {"version": "3", "services": {}}
    all_networks = set()
    all_volumes = set() # Track volumes defined in services

    # Keys from containers.json that map directly or with minor transformation
    # to docker-compose service definitions.
    valid_service_keys = [
        "image", "build", "ports", "volumes", "environment", "env_file",
        "depends_on", "mem_limit", "restart", "command", "entrypoint",
        "working_dir", "user", "labels", "dns", "cap_add", "cap_drop",
        "devices", "expose", "extra_hosts", "healthcheck", "logging",
        "privileged", "read_only", "security_opt", "stdin_open", "stop_grace_period",
        "stop_signal", "sysctls", "tmpfs", "tty", "ulimits", "networks"
        # Note: 'networks' needs special handling below.
        # Keys like 'name' and 'conda' are specific to containers.json and handled separately.
    ]

    try:
        for i, config in enumerate(container_configs):
            if not isinstance(config, dict):
                 msg = f"Invalid configuration format for container #{i+1}: Expected a dictionary."
                 logger.error(msg)
                 return None, msg

            if "name" not in config:
                msg = f"Missing required key 'name' in container configuration #{i+1}."
                logger.error(msg)
                return None, msg

            service_name = config["name"]
            service_definition: Dict[str, Any] = {}

            # Process standard docker-compose keys
            for key in valid_service_keys:
                if key in config:
                    if key == "networks":
                        # Handle networks structure
                        if isinstance(config["networks"], dict):
                            service_networks = {}
                            for net_name, net_config in config["networks"].items():
                                # net_config might contain aliases etc.
                                service_networks[net_name] = net_config or {} # Ensure it's at least an empty dict if null
                                all_networks.add(net_name)
                            service_definition["networks"] = service_networks
                        elif isinstance(config["networks"], list): # Support simple list format
                             service_definition["networks"] = config["networks"]
                             all_networks.update(config["networks"])
                        else:
                            logger.warning(f"Invalid 'networks' format for service '{service_name}'. Skipping networks.")
                    elif key == "volumes":
                         # Track named volumes defined in services
                         service_volumes = config[key]
                         service_definition[key] = service_volumes
                         if isinstance(service_volumes, list):
                             for vol in service_volumes:
                                 if isinstance(vol, str) and ':' in vol:
                                     vol_name = vol.split(':')[0]
                                     # Basic check to see if it looks like a named volume
                                     if not vol_name.startswith(('./', '/')):
                                         all_volumes.add(vol_name)
                         elif isinstance(service_volumes, dict): # Short syntax volumes
                             # Less common, but handle basic case
                             for vol_name in service_volumes.keys():
                                 if not vol_name.startswith(('./', '/')):
                                     all_volumes.add(vol_name)

                    else:
                        service_definition[key] = config[key]

            if not service_definition:
                 logger.warning(f"Service '{service_name}' resulted in an empty definition. Skipping.")
                 continue

            compose_data["services"][service_name] = service_definition

        # Add top-level networks based on collected network names
        if all_networks:
            # Define networks simply; assumes they are externally managed or default bridge
            # Adjust if specific driver/external flags are needed based on conventions
            compose_data["networks"] = {net: None for net in sorted(list(all_networks))}

        # Add top-level volumes based on collected volume names
        if all_volumes:
             # Define volumes simply; assumes default local driver
             compose_data["volumes"] = {vol: None for vol in sorted(list(all_volumes))}

        if not compose_data["services"]:
            msg = "No services were generated from the configuration."
            logger.warning(msg)
            # Return an empty structure, let the caller decide if