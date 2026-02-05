#!/usr/bin/env python3
"""
MLX Inference gRPC Service - Production Implementation

This module provides the production gRPC service for MLX inference.
It integrates with the existing MLX runner implementations in the Geist codebase.
"""

import os
import sys
import time
import uuid
import signal
import logging
import argparse
import json
from concurrent import futures
from typing import Dict, Optional, Iterator, Any

import grpc

# Add parent directory for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
except ImportError:
    psutil = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service name for gRPC
SERVICE_NAME = "geist.inference.v1.InferenceService"


class MLXInferenceRunner:
    """
    MLX inference runner that wraps the existing MLX implementations.

    This class provides a clean interface for model loading and inference,
    supporting both synchronous and streaming generation.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self._loaded = False
        self._quantization: Optional[str] = None

    def load(self, model_id: str, quantization: Optional[str] = None) -> bool:
        """
        Load a model into memory.

        Args:
            model_id: HuggingFace model ID or local path
            quantization: Optional quantization level (4bit, 8bit, etc.)

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model: {model_id}")
            start_time = time.time()

            # Try to use mlx-lm for loading
            try:
                import mlx_lm
                self.model, self.tokenizer = mlx_lm.load(model_id)
            except ImportError:
                # Fallback to existing project implementation
                try:
                    from agents.architectures.llama.llama_mlx import LlamaMLX
                    runner = LlamaMLX()
                    runner.load_model(model_id)
                    self.model = runner.model
                    self.tokenizer = runner.tokenizer
                except ImportError:
                    logger.error("Neither mlx-lm nor project LlamaMLX available")
                    return False

            self.model_id = model_id
            self._loaded = True
            self._quantization = quantization

            load_time = time.time() - start_time
            logger.info(f"Model {model_id} loaded in {load_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False

    def unload(self) -> bool:
        """Unload the model from memory."""
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self._loaded = False
        self._quantization = None

        # Force garbage collection
        import gc
        gc.collect()

        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

        logger.info("Model unloaded")
        return True

    @property
    def loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._loaded

    def _format_prompt(
        self,
        user_prompt: str,
        system_prompt: str = "",
        chat_history: list = None
    ) -> str:
        """Format the prompt using chat template if available."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        messages.append({"role": "user", "content": user_prompt})

        # Try to use tokenizer's chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback to simple format
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        for msg in chat_history or []:
            role = msg.get("role", "user").capitalize()
            parts.append(f"{role}: {msg.get('content', '')}\n")
        parts.append(f"User: {user_prompt}\n")
        parts.append("Assistant:")

        return "".join(parts)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        chat_history: list = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: list = None,
    ) -> Dict[str, Any]:
        """
        Generate a completion.

        Args:
            user_prompt: The user's input
            system_prompt: Optional system instructions
            chat_history: Previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation

        Returns:
            Dictionary with content, finish_reason, and usage stats
        """
        if not self._loaded:
            raise RuntimeError("No model loaded")

        formatted_prompt = self._format_prompt(
            user_prompt, system_prompt, chat_history
        )

        try:
            import mlx_lm

            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            )

            # Calculate token counts
            prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
            completion_tokens = len(self.tokenizer.encode(response))

            return {
                "content": response,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }

        except ImportError:
            # Fallback to existing implementation
            from agents.architectures.llama.llama_mlx import LlamaMLX

            result = LlamaMLX.generate_text(
                self.model,
                self.tokenizer,
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            return {
                "content": result,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            }

    def stream_generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        chat_history: list = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: list = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream tokens as they are generated.

        Yields:
            Dictionaries with token, index, is_final, and optionally finish_reason
        """
        if not self._loaded:
            raise RuntimeError("No model loaded")

        formatted_prompt = self._format_prompt(
            user_prompt, system_prompt, chat_history
        )

        try:
            import mlx_lm

            index = 0
            for token in mlx_lm.stream_generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                yield {
                    "token": token,
                    "index": index,
                    "is_final": False,
                }
                index += 1

            # Final token
            yield {
                "token": "",
                "index": index,
                "is_final": True,
                "finish_reason": "stop",
            }

        except ImportError:
            # Fallback - generate all at once and yield
            result = self.generate(
                user_prompt, system_prompt, chat_history,
                max_tokens, temperature, top_p, stop_sequences
            )
            yield {
                "token": result["content"],
                "index": 0,
                "is_final": True,
                "finish_reason": "stop",
            }


class InferenceService:
    """
    gRPC service implementation for MLX inference.

    This class handles all gRPC method implementations and manages
    the lifecycle of model runners.
    """

    def __init__(self):
        self.runners: Dict[str, MLXInferenceRunner] = {}
        self.default_runner: Optional[MLXInferenceRunner] = None
        self.start_time = time.time()
        self.version = "1.0.0"

    def _get_runner(self, model_id: str) -> MLXInferenceRunner:
        """Get the runner for a given model ID."""
        if model_id in self.runners:
            return self.runners[model_id]
        if self.default_runner and self.default_runner.loaded:
            return self.default_runner
        raise RuntimeError(f"Model {model_id} not loaded")

    def _parse_request(self, request_bytes: bytes) -> Dict[str, Any]:
        """Parse a JSON request."""
        return json.loads(request_bytes.decode('utf-8'))

    def _serialize_response(self, response: Dict[str, Any]) -> bytes:
        """Serialize a response to JSON."""
        return json.dumps(response).encode('utf-8')

    def Complete(self, request_bytes: bytes, context) -> bytes:
        """Handle Complete RPC."""
        try:
            request = self._parse_request(request_bytes)

            runner = self._get_runner(request.get("model_id", ""))
            config = request.get("config", {})

            result = runner.generate(
                user_prompt=request.get("user_prompt", ""),
                system_prompt=request.get("system_prompt", ""),
                chat_history=request.get("chat_history", []),
                max_tokens=config.get("max_tokens", 256),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 1.0),
                stop_sequences=config.get("stop_sequences", []),
            )

            response = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "content": result["content"],
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
                "model_id": request.get("model_id", ""),
                "created": int(time.time()),
            }

            return self._serialize_response(response)

        except Exception as e:
            logger.error(f"Complete error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return self._serialize_response({"error": str(e)})

    def StreamComplete(self, request_bytes: bytes, context) -> Iterator[bytes]:
        """Handle StreamComplete RPC."""
        try:
            request = self._parse_request(request_bytes)

            runner = self._get_runner(request.get("model_id", ""))
            config = request.get("config", {})

            for chunk in runner.stream_generate(
                user_prompt=request.get("user_prompt", ""),
                system_prompt=request.get("system_prompt", ""),
                chat_history=request.get("chat_history", []),
                max_tokens=config.get("max_tokens", 256),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 1.0),
                stop_sequences=config.get("stop_sequences", []),
            ):
                yield self._serialize_response(chunk)

        except Exception as e:
            logger.error(f"StreamComplete error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def LoadModel(self, request_bytes: bytes, context) -> bytes:
        """Handle LoadModel RPC."""
        try:
            request = self._parse_request(request_bytes)
            model_id = request.get("model_id", "")

            start_time = time.time()

            runner = MLXInferenceRunner()
            success = runner.load(
                model_id=model_id,
                quantization=request.get("quantization"),
            )

            if success:
                self.runners[model_id] = runner
                if self.default_runner is None:
                    self.default_runner = runner

            load_time_ms = int((time.time() - start_time) * 1000)
            memory_bytes = psutil.Process().memory_info().rss if psutil else 0

            response = {
                "success": success,
                "message": "Model loaded" if success else "Failed to load model",
                "load_time_ms": load_time_ms,
                "memory_usage_bytes": memory_bytes,
            }

            return self._serialize_response(response)

        except Exception as e:
            logger.error(f"LoadModel error: {e}")
            return self._serialize_response({
                "success": False,
                "message": str(e),
            })

    def UnloadModel(self, request_bytes: bytes, context) -> bytes:
        """Handle UnloadModel RPC."""
        try:
            request = self._parse_request(request_bytes)
            model_id = request.get("model_id", "")

            if model_id in self.runners:
                self.runners[model_id].unload()
                del self.runners[model_id]

                if self.default_runner and self.default_runner.model_id == model_id:
                    self.default_runner = None
                    if self.runners:
                        self.default_runner = next(iter(self.runners.values()))

            return self._serialize_response({
                "success": True,
                "message": "Model unloaded",
            })

        except Exception as e:
            logger.error(f"UnloadModel error: {e}")
            return self._serialize_response({
                "success": False,
                "message": str(e),
            })

    def ListModels(self, request_bytes: bytes, context) -> bytes:
        """Handle ListModels RPC."""
        models = []
        for model_id, runner in self.runners.items():
            models.append({
                "id": model_id,
                "name": model_id.split("/")[-1],
                "loaded": runner.loaded,
                "context_window": 131072,
                "supports_streaming": True,
                "family": "llama" if "llama" in model_id.lower() else "unknown",
            })

        return self._serialize_response({"models": models})

    def GetModelInfo(self, request_bytes: bytes, context) -> bytes:
        """Handle GetModelInfo RPC."""
        request = self._parse_request(request_bytes)
        model_id = request.get("model_id", "")

        if model_id in self.runners:
            runner = self.runners[model_id]
            return self._serialize_response({
                "model": {
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "loaded": runner.loaded,
                    "context_window": 131072,
                    "supports_streaming": True,
                },
                "exists": True,
            })

        return self._serialize_response({"exists": False})

    def Health(self, request_bytes: bytes, context) -> bytes:
        """Handle Health RPC."""
        memory_bytes = psutil.Process().memory_info().rss if psutil else 0
        uptime_seconds = int(time.time() - self.start_time)

        return self._serialize_response({
            "status": "SERVING",
            "version": self.version,
            "loaded_models": len(self.runners),
            "uptime_seconds": uptime_seconds,
            "memory_usage_bytes": memory_bytes,
        })


def create_server(port: int, service: InferenceService) -> grpc.Server:
    """Create and configure the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )

    # Register handlers
    method_handlers = {
        f'/{SERVICE_NAME}/Complete': grpc.unary_unary_rpc_method_handler(
            service.Complete,
        ),
        f'/{SERVICE_NAME}/StreamComplete': grpc.unary_stream_rpc_method_handler(
            service.StreamComplete,
        ),
        f'/{SERVICE_NAME}/LoadModel': grpc.unary_unary_rpc_method_handler(
            service.LoadModel,
        ),
        f'/{SERVICE_NAME}/UnloadModel': grpc.unary_unary_rpc_method_handler(
            service.UnloadModel,
        ),
        f'/{SERVICE_NAME}/ListModels': grpc.unary_unary_rpc_method_handler(
            service.ListModels,
        ),
        f'/{SERVICE_NAME}/GetModelInfo': grpc.unary_unary_rpc_method_handler(
            service.GetModelInfo,
        ),
        f'/{SERVICE_NAME}/Health': grpc.unary_unary_rpc_method_handler(
            service.Health,
        ),
    }

    handler = grpc.method_handlers_generic_handler(
        SERVICE_NAME, method_handlers
    )
    server.add_generic_rpc_handlers((handler,))
    server.add_insecure_port(f'[::]:{port}')

    return server


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLX Inference gRPC Service")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--model", type=str, help="Default model to load on startup")
    args = parser.parse_args()

    service = InferenceService()

    # Load default model if specified
    if args.model:
        logger.info(f"Loading default model: {args.model}")
        request = json.dumps({"model_id": args.model}).encode('utf-8')
        service.LoadModel(request, None)

    server = create_server(args.port, service)
    server.start()

    logger.info(f"MLX Inference Service running on port {args.port}")

    # Handle shutdown
    stop_event = futures.Future()

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        stop_event.set_result(None)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        stop_event.result()
    except KeyboardInterrupt:
        pass

    server.stop(grace=5)
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
