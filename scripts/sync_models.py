#!/usr/bin/env python3
"""
Model Synchronization Script

Fetches available models from Anthropic, OpenAI, and HuggingFace APIs
and updates the model registry with newly discovered models.

Environment Variables:
    ANTHROPIC_API_KEY: API key for Anthropic
    OPENAI_API_KEY: API key for OpenAI
    HUGGINGFACE_API_KEY: API key for HuggingFace (optional)

Usage:
    python scripts/sync_models.py --dry-run
    python scripts/sync_models.py --update-backend
    python scripts/sync_models.py --update-all
    python scripts/sync_models.py --provider openai --verbose
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from scripts.model_filter_config import (
    should_include_model,
    get_model_metadata,
    generate_display_name,
    HUGGINGFACE_MODELS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class DiscoveredModel:
    """Represents a model discovered from an API."""
    id: str
    name: str
    provider: str
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = True
    recommended: bool = False
    family: Optional[str] = None


def fetch_openai_models(api_key: str) -> List[DiscoveredModel]:
    """
    Fetch available models from OpenAI API.

    Args:
        api_key: OpenAI API key

    Returns:
        List of discovered models
    """
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        models = []
        for model_data in data.get("data", []):
            model_id = model_data.get("id", "")

            # Filter for chat models
            if not should_include_model(model_id):
                continue

            # Get metadata overrides
            metadata = get_model_metadata(model_id) or {}

            models.append(DiscoveredModel(
                id=model_id,
                name=metadata.get("name", generate_display_name(model_id)),
                provider="openai",
                context_window=metadata.get("context_window"),
                max_output_tokens=metadata.get("max_output_tokens"),
                supports_vision=metadata.get("supports_vision", False),
                supports_function_calling=metadata.get("supports_function_calling", True),
                supports_streaming=True,
                recommended=metadata.get("recommended", False),
                family=metadata.get("family"),
            ))

        logger.info(f"Fetched {len(models)} OpenAI models")
        return models

    except requests.RequestException as e:
        logger.error(f"Failed to fetch OpenAI models: {e}")
        return []


def fetch_anthropic_models(api_key: str) -> List[DiscoveredModel]:
    """
    Fetch available models from Anthropic API.

    Args:
        api_key: Anthropic API key

    Returns:
        List of discovered models
    """
    try:
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        models = []
        for model_data in data.get("data", []):
            model_id = model_data.get("id", "")

            # Get metadata overrides
            metadata = get_model_metadata(model_id) or {}

            # Extract display name from model data or generate
            display_name = model_data.get("display_name") or metadata.get("name") or generate_display_name(model_id)

            models.append(DiscoveredModel(
                id=model_id,
                name=display_name,
                provider="anthropic",
                context_window=metadata.get("context_window", 200000),
                max_output_tokens=metadata.get("max_output_tokens", 4096),
                supports_vision=metadata.get("supports_vision", True),
                supports_function_calling=metadata.get("supports_function_calling", True),
                supports_streaming=True,
                recommended=metadata.get("recommended", True),
                family=metadata.get("family", "claude"),
            ))

        logger.info(f"Fetched {len(models)} Anthropic models")
        return models

    except requests.RequestException as e:
        logger.error(f"Failed to fetch Anthropic models: {e}")
        return []


def fetch_huggingface_models(api_key: Optional[str] = None) -> List[DiscoveredModel]:
    """
    Fetch recommended models from HuggingFace.

    Args:
        api_key: Optional HuggingFace API key

    Returns:
        List of discovered models
    """
    models = []

    for model_id in HUGGINGFACE_MODELS:
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.get(
                f"https://huggingface.co/api/models/{model_id}",
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Generate display name
                name_parts = model_id.split("/")
                display_name = name_parts[-1] if len(name_parts) > 1 else model_id

                models.append(DiscoveredModel(
                    id=model_id,
                    name=display_name.replace("-", " ").title(),
                    provider="huggingface",
                    context_window=None,  # HuggingFace doesn't always provide this
                    max_output_tokens=None,
                    supports_vision=False,
                    supports_function_calling=False,
                    supports_streaming=True,
                    recommended=True,
                    family=name_parts[0] if len(name_parts) > 1 else None,
                ))
            else:
                logger.warning(f"Could not fetch HuggingFace model {model_id}: {response.status_code}")

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch HuggingFace model {model_id}: {e}")

    logger.info(f"Fetched {len(models)} HuggingFace models")
    return models


def generate_model_registry_code(models_by_provider: Dict[str, List[DiscoveredModel]]) -> str:
    """
    Generate Python code for DISCOVERED_MODELS.

    Args:
        models_by_provider: Dictionary mapping provider to list of models

    Returns:
        Python code string
    """
    lines = [
        "# Auto-generated by scripts/sync_models.py",
        "# Do not edit manually",
        "",
        "DISCOVERED_MODELS: Dict[OnlineModelProviders, List[ModelInfo]] = {",
    ]

    provider_map = {
        "openai": "OnlineModelProviders.OPENAI",
        "anthropic": "OnlineModelProviders.ANTHROPIC",
        "huggingface": "OnlineModelProviders.HUGGINGFACE",
        "xai": "OnlineModelProviders.XAI",
        "groq": "OnlineModelProviders.GROQ",
    }

    for provider, models in models_by_provider.items():
        if not models:
            continue

        provider_enum = provider_map.get(provider, f'OnlineModelProviders.{provider.upper()}')
        lines.append(f"    {provider_enum}: [")

        for model in models:
            lines.append(f"        ModelInfo(")
            lines.append(f'            id="{model.id}",')
            lines.append(f'            name="{model.name}",')
            lines.append(f"            provider={provider_enum},")
            if model.context_window:
                lines.append(f"            context_window={model.context_window},")
            if model.max_output_tokens:
                lines.append(f"            max_output_tokens={model.max_output_tokens},")
            lines.append(f"            supports_vision={model.supports_vision},")
            lines.append(f"            supports_function_calling={model.supports_function_calling},")
            lines.append(f"            supports_streaming={model.supports_streaming},")
            lines.append(f"            recommended={model.recommended},")
            if model.family:
                lines.append(f'            family="{model.family}",')
            lines.append(f"        ),")

        lines.append("    ],")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def update_backend_registry(models_by_provider: Dict[str, List[DiscoveredModel]], dry_run: bool = False) -> None:
    """
    Update agents/architectures/registry.py with discovered models.

    Args:
        models_by_provider: Dictionary mapping provider to list of models
        dry_run: If True, only show what would be changed
    """
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agents", "architectures", "registry.py"
    )

    code = generate_model_registry_code(models_by_provider)

    if dry_run:
        print("\n--- Generated DISCOVERED_MODELS code ---")
        print(code)
        print("--- End of generated code ---\n")
        return

    # Read existing file
    with open(registry_path, "r") as f:
        content = f.read()

    # Find and replace DISCOVERED_MODELS section
    import re
    pattern = r"# Dynamic registry - populated by sync script\n# Auto-generated - do not edit manually\nDISCOVERED_MODELS: Dict\[OnlineModelProviders, List\[ModelInfo\]\] = \{\}"

    replacement = f"""# Dynamic registry - populated by sync script
{code.strip()}"""

    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        with open(registry_path, "w") as f:
            f.write(new_content)
        logger.info(f"Updated {registry_path}")
    else:
        logger.warning("Could not find DISCOVERED_MODELS section to update")


def print_model_summary(models_by_provider: Dict[str, List[DiscoveredModel]], verbose: bool = False) -> None:
    """
    Print a summary of discovered models.

    Args:
        models_by_provider: Dictionary mapping provider to list of models
        verbose: If True, print detailed information
    """
    print("\n=== Model Discovery Summary ===\n")

    total = 0
    for provider, models in sorted(models_by_provider.items()):
        print(f"{provider.upper()}: {len(models)} models")
        total += len(models)

        if verbose:
            for model in models:
                print(f"  - {model.id}")
                print(f"    Name: {model.name}")
                if model.context_window:
                    print(f"    Context: {model.context_window:,}")
                if model.supports_vision:
                    print(f"    Vision: Yes")
                if model.recommended:
                    print(f"    Recommended: Yes")
            print()

    print(f"\nTotal: {total} models\n")


def check_missing_api_keys(provider: str = "all") -> List[str]:
    """
    Check for missing API keys for online providers.

    Args:
        provider: Which provider to check ("openai", "anthropic", "huggingface", or "all")

    Returns:
        List of missing API key names
    """
    missing_keys = []

    # Define required keys for each provider
    key_checks = {
        "openai": ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        "anthropic": ("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY")),
    }

    # Check which providers need keys
    providers_to_check = list(key_checks.keys()) if provider == "all" else [provider]

    for prov in providers_to_check:
        if prov in key_checks:
            key_name, key_value = key_checks[prov]
            if not key_value:
                missing_keys.append(key_name)

    return missing_keys


def print_missing_keys_warning(missing_keys: List[str]) -> None:
    """
    Print a prominent warning about missing API keys.

    Args:
        missing_keys: List of missing API key environment variable names
    """
    if not missing_keys:
        return

    warning_lines = [
        "",
        "=" * 60,
        "⚠️  WARNING: Missing API Keys for Online Providers",
        "=" * 60,
        "",
        "The following API keys are not set in your environment:",
    ]
    for key in missing_keys:
        warning_lines.append(f"  • {key}")
    warning_lines.extend([
        "",
        "Models from these providers will NOT be synced.",
        "Set these environment variables to enable syncing:",
    ])
    for key in missing_keys:
        warning_lines.append(f"  export {key}=your_key_here")
    warning_lines.extend(["=" * 60, ""])

    # Print to stderr to match logger output ordering
    for line in warning_lines:
        print(line, file=sys.stderr)


def main():
    """Main entry point for the sync script."""
    parser = argparse.ArgumentParser(
        description="Sync available models from provider APIs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing"
    )
    parser.add_argument(
        "--update-backend",
        action="store_true",
        help="Update backend registry"
    )
    parser.add_argument(
        "--update-all",
        action="store_true",
        help="Update both backend and frontend"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "huggingface", "all"],
        default="all",
        help="Sync specific provider (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed model info"
    )
    parser.add_argument(
        "--json-out",
        type=str,
        help="Optional path to write JSON output"
    )

    args = parser.parse_args()

    # Check for missing API keys and warn user
    missing_keys = check_missing_api_keys(args.provider)
    print_missing_keys_warning(missing_keys)

    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    huggingface_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

    models_by_provider: Dict[str, List[DiscoveredModel]] = {}

    # Fetch models based on provider selection
    if args.provider in ["openai", "all"]:
        if openai_key:
            models_by_provider["openai"] = fetch_openai_models(openai_key)
        else:
            logger.warning("OPENAI_API_KEY not set, skipping OpenAI")

    if args.provider in ["anthropic", "all"]:
        if anthropic_key:
            models_by_provider["anthropic"] = fetch_anthropic_models(anthropic_key)
        else:
            logger.warning("ANTHROPIC_API_KEY not set, skipping Anthropic")

    if args.provider in ["huggingface", "all"]:
        models_by_provider["huggingface"] = fetch_huggingface_models(huggingface_key)

    # Print summary
    print_model_summary(models_by_provider, verbose=args.verbose)

    # Write JSON output if requested
    if args.json_out:
        output = {
            provider: [asdict(m) for m in models]
            for provider, models in models_by_provider.items()
        }
        with open(args.json_out, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Wrote JSON output to {args.json_out}")

    # Update backend if requested
    if args.update_backend or args.update_all:
        update_backend_registry(models_by_provider, dry_run=args.dry_run)
    elif args.dry_run:
        update_backend_registry(models_by_provider, dry_run=True)


if __name__ == "__main__":
    main()
