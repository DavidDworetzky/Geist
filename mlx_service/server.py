#!/usr/bin/env python3
"""
MLX Inference gRPC Service

This service provides gRPC-based inference using MLX for Apple Silicon.
It wraps the existing MLX runner implementations and exposes them via gRPC.
"""

import os
import sys
import time
import uuid
import signal
import logging
import argparse
import psutil
from concurrent import futures
from typing import Dict, Optional, Iterator

import grpc
from grpc_reflection.v1alpha import reflection

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_service import inference_pb2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLXRunner:
    """Wrapper around MLX inference capabilities."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self._loaded = False

    def load(self, model_id: str, quantization: str = None) -> bool:
        """Load a model into memory."""
        try:
            import mlx_lm

            logger.info(f"Loading model: {model_id}")
            start_time = time.time()

            self.model, self.tokenizer = mlx_lm.load(model_id)
            self.model_id = model_id
            self._loaded = True

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def unload(self) -> bool:
        """Unload the model from memory."""
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self._loaded = False
        return True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: list = None,
    ) -> Dict:
        """Generate a completion."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        import mlx_lm

        # Build messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # Generate
        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )

        # Count tokens
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

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: list = None,
    ) -> Iterator[Dict]:
        """Stream tokens as they are generated."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        import mlx_lm

        # Build messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # Stream generation
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


class InferenceServicer:
    """gRPC service implementation for inference."""

    SERVICE_NAME = "geist.inference.v1.InferenceService"

    def __init__(self):
        self.runners: Dict[str, MLXRunner] = {}
        self.default_runner: Optional[MLXRunner] = None
        self.start_time = time.time()
        self.version = "1.0.0"

    def _get_runner(self, model_id: str) -> MLXRunner:
        """Get or create a runner for the given model."""
        if model_id in self.runners:
            return self.runners[model_id]
        if self.default_runner and self.default_runner.loaded:
            return self.default_runner
        raise RuntimeError(f"Model {model_id} not loaded")

    def Complete(self, request, context) -> inference_pb2.CompletionResponse:
        """Synchronous completion."""
        try:
            runner = self._get_runner(request.model_id)

            config = request.config or inference_pb2.GenerationConfig()

            result = runner.generate(
                prompt=request.user_prompt,
                system_prompt=request.system_prompt,
                max_tokens=config.max_tokens or 256,
                temperature=config.temperature or 0.7,
                top_p=config.top_p or 1.0,
                stop_sequences=list(config.stop_sequences) if config.stop_sequences else None,
            )

            return inference_pb2.CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                content=result["content"],
                finish_reason=result["finish_reason"],
                usage=inference_pb2.UsageStats(
                    prompt_tokens=result["usage"]["prompt_tokens"],
                    completion_tokens=result["usage"]["completion_tokens"],
                    total_tokens=result["usage"]["total_tokens"],
                ),
                model_id=request.model_id,
                created=int(time.time()),
            )
        except Exception as e:
            logger.error(f"Completion error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.CompletionResponse()

    def StreamComplete(self, request, context) -> Iterator[inference_pb2.TokenChunk]:
        """Streaming completion."""
        try:
            runner = self._get_runner(request.model_id)

            config = request.config or inference_pb2.GenerationConfig()

            for chunk in runner.stream_generate(
                prompt=request.user_prompt,
                system_prompt=request.system_prompt,
                max_tokens=config.max_tokens or 256,
                temperature=config.temperature or 0.7,
                top_p=config.top_p or 1.0,
                stop_sequences=list(config.stop_sequences) if config.stop_sequences else None,
            ):
                yield inference_pb2.TokenChunk(
                    token=chunk["token"],
                    is_final=chunk["is_final"],
                    finish_reason=chunk.get("finish_reason", ""),
                    index=chunk["index"],
                )
        except Exception as e:
            logger.error(f"Stream error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def LoadModel(self, request, context) -> inference_pb2.LoadModelResponse:
        """Load a model into memory."""
        try:
            start_time = time.time()

            runner = MLXRunner()
            success = runner.load(
                model_id=request.model_id,
                quantization=request.quantization or None,
            )

            if success:
                self.runners[request.model_id] = runner
                if self.default_runner is None:
                    self.default_runner = runner

            load_time_ms = int((time.time() - start_time) * 1000)
            memory_bytes = psutil.Process().memory_info().rss

            return inference_pb2.LoadModelResponse(
                success=success,
                message="Model loaded successfully" if success else "Failed to load model",
                load_time_ms=load_time_ms,
                memory_usage_bytes=memory_bytes,
            )
        except Exception as e:
            logger.error(f"Load model error: {e}")
            return inference_pb2.LoadModelResponse(
                success=False,
                message=str(e),
            )

    def UnloadModel(self, request, context) -> inference_pb2.UnloadModelResponse:
        """Unload a model from memory."""
        try:
            if request.model_id in self.runners:
                self.runners[request.model_id].unload()
                del self.runners[request.model_id]

                if self.default_runner and self.default_runner.model_id == request.model_id:
                    self.default_runner = None
                    if self.runners:
                        self.default_runner = next(iter(self.runners.values()))

            return inference_pb2.UnloadModelResponse(
                success=True,
                message="Model unloaded successfully",
            )
        except Exception as e:
            logger.error(f"Unload model error: {e}")
            return inference_pb2.UnloadModelResponse(
                success=False,
                message=str(e),
            )

    def ListModels(self, request, context) -> inference_pb2.ListModelsResponse:
        """List available models."""
        models = []
        for model_id, runner in self.runners.items():
            models.append(inference_pb2.ModelInfo(
                id=model_id,
                name=model_id.split("/")[-1],
                loaded=runner.loaded,
                context_window=131072,  # Default for Llama 3.1
                supports_streaming=True,
                family="llama" if "llama" in model_id.lower() else "unknown",
            ))

        return inference_pb2.ListModelsResponse(models=models)

    def GetModelInfo(self, request, context) -> inference_pb2.GetModelInfoResponse:
        """Get information about a specific model."""
        if request.model_id in self.runners:
            runner = self.runners[request.model_id]
            return inference_pb2.GetModelInfoResponse(
                model=inference_pb2.ModelInfo(
                    id=request.model_id,
                    name=request.model_id.split("/")[-1],
                    loaded=runner.loaded,
                    context_window=131072,
                    supports_streaming=True,
                ),
                exists=True,
            )
        return inference_pb2.GetModelInfoResponse(exists=False)

    def Health(self, request, context) -> inference_pb2.HealthResponse:
        """Health check."""
        memory_bytes = psutil.Process().memory_info().rss
        uptime_seconds = int(time.time() - self.start_time)

        return inference_pb2.HealthResponse(
            status="SERVING",
            version=self.version,
            loaded_models=len(self.runners),
            uptime_seconds=uptime_seconds,
            memory_usage_bytes=memory_bytes,
        )


def create_grpc_server(servicer: InferenceServicer, port: int) -> grpc.Server:
    """Create and configure the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
    )

    # Register service methods manually since we don't have generated code
    # In production, use add_InferenceServiceServicer_to_server from generated code

    # For now, we'll use a simple approach
    method_handlers = {
        '/geist.inference.v1.InferenceService/Complete': grpc.unary_unary_rpc_method_handler(
            servicer.Complete,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
        '/geist.inference.v1.InferenceService/StreamComplete': grpc.unary_stream_rpc_method_handler(
            servicer.StreamComplete,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
        '/geist.inference.v1.InferenceService/LoadModel': grpc.unary_unary_rpc_method_handler(
            servicer.LoadModel,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
        '/geist.inference.v1.InferenceService/UnloadModel': grpc.unary_unary_rpc_method_handler(
            servicer.UnloadModel,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
        '/geist.inference.v1.InferenceService/ListModels': grpc.unary_unary_rpc_method_handler(
            servicer.ListModels,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
        '/geist.inference.v1.InferenceService/Health': grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
        '/geist.inference.v1.InferenceService/GetModelInfo': grpc.unary_unary_rpc_method_handler(
            servicer.GetModelInfo,
            request_deserializer=lambda x: x,
            response_serializer=lambda x: x,
        ),
    }

    generic_handler = grpc.method_handlers_generic_handler(
        'geist.inference.v1.InferenceService',
        method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))

    server.add_insecure_port(f'[::]:{port}')

    return server


def serve(port: int = 50051, default_model: str = None):
    """Start the gRPC server."""
    servicer = InferenceServicer()

    # Optionally load default model
    if default_model:
        logger.info(f"Loading default model: {default_model}")
        request = inference_pb2.LoadModelRequest(model_id=default_model)
        servicer.LoadModel(request, None)

    server = create_grpc_server(servicer, port)
    server.start()

    logger.info(f"MLX Inference Service running on port {port}")

    # Handle shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(grace=5)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description="MLX Inference gRPC Service")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--model", type=str, default=None, help="Default model to load")
    args = parser.parse_args()

    serve(port=args.port, default_model=args.model)


if __name__ == "__main__":
    main()
