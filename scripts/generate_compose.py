#!/usr/bin/env python3
"""
Script to generate docker-compose.yml from a JSON configuration file,
potentially merging with an existing docker-compose.yml.

This script reads container definitions from a JSON file (e.g., containers.json),
loads an existing docker-compose.yml file as a base, generates new services
based on a template ('backend'), and merges configurations before writing
the result back to docker-compose.yml.
"""

import json
import logging
import argparse
import copy 
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml 
reserved_source_keys = ["backend", "db", "frontend", "adapter"]
template_name = "adapter"

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

def load_base_compose(compose_path: Path) -> Tuple[Dict[str, Any],str]:
    """
    Loads the base docker-compose.yml file.

    Args:
        compose_path: Path to the docker-compose.yml file.

    Returns:
        Tuple containing:
            - The loaded docker-compose data (dict). Returns a default structure if not found.
            - Message describing the result (str).
    """
    # Only include essential default keys
    base_keys = ["version", "services"]
    
    if not compose_path.is_file():
        msg = f"Base docker-compose file not found: {compose_path}. Starting with empty config."
        logger.warning(msg)
        # Return minimal default structure
        return {"version": "3", "services": {}}, msg
    try:
        with open(compose_path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            msg = f"Invalid format in {compose_path}: Expected a dictionary."
            logger.error(msg)
            # Return minimal default structure
            return {"version": "3", "services": {}}, msg
            
        # Only ensure essential keys exist
        if "version" not in data:
            data["version"] = "3"
        if "services" not in data:
            data["services"] = {}
            
        logger.info(f"Successfully loaded base configuration from {compose_path}")
        return data, f"Successfully loaded base configuration from {compose_path}"
    except yaml.YAMLError as e:
        msg = f"Error decoding YAML from {compose_path}: {e}"
        logger.error(msg)
        return {"version": "3", "services": {}}, msg
    except Exception as e:
        msg = f"An unexpected error occurred while reading {compose_path}: {e}"
        logger.error(msg, exc_info=True)
        return {"version": "3", "services": {}}, msg


def generate_compose_dict(
    container_configs: List[Dict[str, Any]],
    base_compose_path: Path
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Generates the docker-compose dictionary structure by merging container
    configurations with a base docker-compose file and generating new services
    from a template.

    Args:
        container_configs: A list of dictionaries, each representing a service
                           configuration from the input JSON.
        base_compose_path: Path to the base docker-compose.yml file.

    Returns:
        Tuple containing:
            - The generated docker-compose dictionary or None if error.
            - Message describing the result (str).
    """
    # Load the base docker-compose file
    compose_data, load_msg = load_base_compose(base_compose_path)
    logger.info(load_msg)

    # Find the template configuration from containers.json
    template_config = next((c for c in container_configs if c.get("name") == template_name), None)

    # Get the template service definition from the template config
    template_service_definition = (next((x for x in container_configs if x.get("name") == template_name), None))

    if not template_service_definition:
        msg = f"Template service '{template_name}' not found in base {base_compose_path}. Cannot generate derived services."
        raise ValueError(msg)

    all_networks = set(compose_data.get("networks", {}).keys())
    all_volumes = set(compose_data.get("volumes", {}).keys()) # Track volumes defined in services

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

    processed_configs = set() # Keep track of configs used

    try:
        for i, config in enumerate(container_configs):
            if not isinstance(config, dict):
                 msg = f"Invalid configuration format for container #{i+1}: Expected a dictionary."
                 logger.error(msg)
                 continue # Skip invalid entry

            if "name" not in config:
                msg = f"Missing required key 'name' in container configuration #{i+1}."
                logger.error(msg)
                continue # Skip invalid entry

            service_name = config["name"]
            processed_configs.add(service_name)
            service_definition: Dict[str, Any] = {}

            # Determine if this is a reserved key or a derived service
            is_reserved = service_name in reserved_source_keys
            is_derived = not is_reserved and template_service_definition is not None

            if is_reserved:
                # Generate definition from scratch based on config, potentially overwriting base
                target_service_name = service_name
                # Start with an empty definition, it will overwrite if exists
                current_service_definition = {}
            elif is_derived:
                # Generate definition by copying template and overriding
                target_service_name = f"adapter-{service_name}"
                # Start with a deep copy of the template definition
                current_service_definition = copy.deepcopy(template_service_definition)
                logger.info(f"Generating service '{target_service_name}' based on template '{template_name}'.")
            else:
                # Not reserved, and template is missing or invalid - skip derivation
                if not is_reserved:
                    logger.warning(f"Skipping generation for '{service_name}' as template '{template_name}' is missing in base compose file.")
                continue # Skip to next config

            # Add the processed/generated service definition to compose_data
            if not current_service_definition:
                 logger.warning(f"Configuration for '{service_name}' resulted in an empty service definition for '{target_service_name}'. Skipping.")
                 continue

            compose_data["services"][target_service_name] = current_service_definition


        # Add/Update top-level networks based on collected network names
        # Ensure existing network definitions (like external=true) are preserved if not overridden
        final_networks = compose_data.get("networks", {}) # Start with base networks
        for net in sorted(list(all_networks)):
            final_networks.setdefault(net, None) # Add if not present, keep existing config if present
        if final_networks:
            compose_data["networks"] = final_networks
        elif "networks" in compose_data:
             del compose_data["networks"] # Clean up if no networks ended up being used


        # Add/Update top-level volumes based on collected volume names
        # Ensure existing volume definitions are preserved
        final_volumes = compose_data.get("volumes", {}) # Start with base volumes
        for vol in sorted(list(all_volumes)):
             final_volumes.setdefault(vol, None) # Add if not present, keep existing config if present
        if final_volumes:
            compose_data["volumes"] = final_volumes
        elif "volumes" in compose_data:
            del compose_data["volumes"] # Clean up if no volumes ended up being used


        if not compose_data.get("services"):
            msg = "No services were defined in the final configuration."
            logger.warning(msg)
            # Decide if returning None or empty structure is better
            # return None, msg
        else:
            msg = f"Successfully generated docker-compose dictionary. Services: {list(compose_data['services'].keys())}"
            logger.info(msg)

        return compose_data, msg

    except Exception as e:
        msg = f"An unexpected error occurred during compose dictionary generation: {e}"
        logger.error(msg, exc_info=True)
        return None, msg

def write_compose_file(compose_data: Dict[str, Any], output_path: Path) -> Tuple[bool, str]:
    """
    Writes the docker-compose dictionary to a YAML file.
    Ports and env_file entries are explicitly converted to strings to ensure
    they are appropriately represented (and potentially quoted) in the output YAML.

    Args:
        compose_data: The dictionary representing the docker-compose configuration.
        output_path: Path to write the docker-compose.yml file.

    Returns:
        Tuple containing:
            - Boolean indicating success.
            - Message describing the result (str).
    """
    try:
        # Preprocess compose_data to ensure specific fields are strings
        if "services" in compose_data and isinstance(compose_data.get("services"), dict):
            for service_config in compose_data["services"].values():
                if isinstance(service_config, dict):
                    # Ensure ports are strings
                    if "ports" in service_config and isinstance(service_config.get("ports"), list):
                        service_config["ports"] = [
                            str(port) for port in service_config["ports"]
                        ]
                    
                    # Ensure env_file entries are strings
                    if "env_file" in service_config and isinstance(service_config.get("env_file"), list):
                        service_config["env_file"] = [
                            str(env_file_path) for env_file_path in service_config["env_file"]
                        ]
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
        msg = f"Successfully wrote docker-compose configuration to {output_path}"
        logger.info(msg)
        return True, msg
    except Exception as e:
        msg = f"An error occurred while writing to {output_path}: {e}"
        logger.error(msg, exc_info=True)
        return False, msg

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate docker-compose.yml from JSON config, merging with existing file.")
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to the input JSON configuration file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the output docker-compose.yml file (default: {DEFAULT_OUTPUT_PATH})"
    )
    args = parser.parse_args()

    config_path = args.config
    output_path = args.output # This is also the base path now

    # 1. Load container configurations from JSON
    container_configs, msg = load_container_config(config_path)
    if container_configs is None:
        logger.error(f"Failed to load container config: {msg}")
        return

    compose_data, msg = generate_compose_dict(container_configs, output_path)
    if compose_data is None:
        logger.error(f"Failed to generate compose dictionary: {msg}")
        return

    # 3. Write the result back to the output file
    success, msg = write_compose_file(compose_data, output_path)
    if not success:
        logger.error(f"Failed to write compose file: {msg}")
    else:
        logger.info("Docker-compose generation process completed.")


if __name__ == "__main__":
    main()
        