# Go Migration Plan for Geist Server

## Executive Summary

This document outlines the strategy for migrating the Geist server core from Python (FastAPI/Uvicorn) to Go, while maintaining MLX interop for local inference on Apple Silicon. The migration will improve performance, reduce memory footprint, simplify deployment, and enable better concurrency handling.

---

## 1. Current Architecture Overview

### Python Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Routes: /agent/*, /adapter/*, /api/v1/*               ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Agent Layer (BaseAgent, LocalAgent, OnlineAgent)      ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Runner Layer (MLXLlamaRunner, VLLMRunner)             ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  MLX/PyTorch Inference (LlamaMLX, LlamaTransformer)    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Key Components to Migrate
| Component | Python Location | Go Strategy |
|-----------|-----------------|-------------|
| HTTP Server | `app/main.py` (FastAPI) | `net/http` + Chi/Gin router |
| Request Models | `app/models/completion.py` | Go structs with JSON tags |
| Agent Interface | `agents/base_agent.py` | Go interfaces |
| Online Agent | `agents/online_agent.py` | Native Go HTTP client |
| Local Agent | `agents/local_agent.py` | Go + Python subprocess |
| MLX Runner | `agents/architectures/mlx_llama_runner.py` | Python subprocess via gRPC/JSON-RPC |
| Database ORM | SQLAlchemy models | GORM or sqlc |
| Config | `app/environment.py` | Viper or envconfig |

---

## 2. Target Architecture

### Go Server with Python MLX Subprocess
```
┌──────────────────────────────────────────────────────────────────────┐
│                         Go Server (main process)                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  HTTP Router (Chi/Gin)                                         │  │
│  │  Routes: /agent/*, /adapter/*, /api/v1/*                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Agent Manager (Go)                                            │  │
│  │  - OnlineAgent: Native HTTP client to OpenAI/Anthropic/Groq   │  │
│  │  - LocalAgent: gRPC client to MLX subprocess                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Database Layer (GORM/pgx)                                     │  │
│  │  - Chat sessions, Agent presets, User settings                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                               │
                    gRPC/Unix Socket/HTTP
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Python MLX Inference Service                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  gRPC Server (grpcio) / FastAPI minimal                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  MLX Runner (unchanged)                                        │  │
│  │  - LlamaMLX, model loading, tokenization, generation          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. MLX Interop Strategy

### Option A: gRPC Interface (Recommended)

**Pros:**
- Type-safe, well-defined interface via Protocol Buffers
- Excellent Go support (google.golang.org/grpc)
- Streaming support for token-by-token generation
- Language agnostic - can swap Python for Rust/C++ later

**Cons:**
- Additional complexity of proto definitions
- Extra dependency (grpcio in Python)

**Implementation:**

```protobuf
// proto/inference.proto
syntax = "proto3";

package geist.inference;

service InferenceService {
  // Unary completion
  rpc Complete(CompletionRequest) returns (CompletionResponse);

  // Streaming token generation
  rpc StreamComplete(CompletionRequest) returns (stream TokenChunk);

  // Model management
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);

  // Health check
  rpc Health(HealthRequest) returns (HealthResponse);
}

message CompletionRequest {
  string model_id = 1;
  string system_prompt = 2;
  string user_prompt = 3;
  GenerationConfig config = 4;
  repeated Message chat_history = 5;
}

message GenerationConfig {
  int32 max_tokens = 1;
  float temperature = 2;
  float top_p = 3;
  float frequency_penalty = 4;
  float presence_penalty = 5;
  repeated string stop_sequences = 6;
}

message Message {
  string role = 1;
  string content = 2;
}

message CompletionResponse {
  string id = 1;
  string content = 2;
  string finish_reason = 3;
  UsageStats usage = 4;
}

message TokenChunk {
  string token = 1;
  bool is_final = 2;
  string finish_reason = 3;
}

message UsageStats {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  int32 total_tokens = 3;
}
```

### Option B: JSON-RPC over Unix Socket

**Pros:**
- Simpler setup, no proto compilation needed
- Low latency via Unix domain sockets
- Easy debugging (human-readable JSON)

**Cons:**
- No type safety at interface boundary
- Manual schema synchronization

### Option C: HTTP/REST (Simplest)

**Pros:**
- Minimal changes to existing Python code
- Can reuse FastAPI routes
- Easy to test with curl/Postman

**Cons:**
- Higher latency than Unix sockets/gRPC
- No native streaming (requires SSE or WebSocket)

### Recommendation: Option A (gRPC)

gRPC provides the best balance of performance, type safety, and streaming support. The Python inference service becomes a dedicated microservice that can be independently scaled or replaced.

---

## 4. Directory Structure

```
geist/
├── cmd/
│   └── server/
│       └── main.go                 # Entry point
├── internal/
│   ├── api/
│   │   ├── router.go               # HTTP router setup
│   │   ├── middleware/
│   │   │   ├── auth.go
│   │   │   ├── logging.go
│   │   │   └── cors.go
│   │   └── handlers/
│   │       ├── agent.go            # /agent/* handlers
│   │       ├── models.go           # /api/v1/models
│   │       ├── voice.go            # /api/v1/voice
│   │       ├── workflows.go        # /api/v1/workflows
│   │       ├── files.go            # /api/v1/files
│   │       └── user_settings.go    # /api/v1/user-settings
│   ├── agent/
│   │   ├── agent.go                # Agent interface
│   │   ├── online_agent.go         # HTTP-based agents
│   │   ├── local_agent.go          # gRPC client to MLX
│   │   ├── factory.go              # Agent factory
│   │   └── context.go              # Agent context/state
│   ├── inference/
│   │   ├── client.go               # gRPC client wrapper
│   │   ├── proto/                  # Generated protobuf Go code
│   │   │   └── inference.pb.go
│   │   └── pool.go                 # Connection pooling
│   ├── models/
│   │   ├── completion.go           # API request/response models
│   │   ├── agent.go                # Agent domain model
│   │   └── registry.go             # Model registry
│   ├── database/
│   │   ├── database.go             # DB connection
│   │   ├── migrations/             # SQL migrations
│   │   └── repository/
│   │       ├── chat_session.go
│   │       ├── agent_preset.go
│   │       └── user_settings.go
│   ├── adapters/
│   │   ├── registry.go             # Adapter registry
│   │   ├── sendgrid.go
│   │   └── search.go
│   └── config/
│       └── config.go               # Configuration loading
├── pkg/
│   └── httputil/                   # Shared HTTP utilities
├── proto/
│   └── inference.proto             # Protocol buffer definitions
├── mlx_service/                    # Python MLX subprocess
│   ├── __init__.py
│   ├── server.py                   # gRPC server implementation
│   ├── runner.py                   # MLX runner (from existing code)
│   └── requirements.txt
├── migrations/                     # Database migrations (existing)
├── client/                         # React frontend (unchanged)
├── tests/
│   ├── integration/
│   └── unit/
├── go.mod
├── go.sum
├── Makefile
└── Dockerfile
```

---

## 5. Core Go Interfaces

### Agent Interface

```go
// internal/agent/agent.go
package agent

import (
    "context"
    "io"
)

type Agent interface {
    // CompleteText generates a completion for the given prompt
    CompleteText(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

    // StreamCompleteText streams tokens as they're generated
    StreamCompleteText(ctx context.Context, req *CompletionRequest) (<-chan Token, error)

    // Initialize sets up the agent with a task prompt
    Initialize(ctx context.Context, taskPrompt string) error

    // Tick advances the agent's execution loop
    Tick(ctx context.Context) (*TickResult, error)

    // GetChatHistory retrieves the conversation history
    GetChatHistory(ctx context.Context, sessionID string) ([]Message, error)
}

type CompletionRequest struct {
    Prompt            string   `json:"prompt"`
    SystemPrompt      string   `json:"system_prompt,omitempty"`
    MaxTokens         int      `json:"max_tokens"`
    Temperature       float64  `json:"temperature"`
    TopP              float64  `json:"top_p"`
    FrequencyPenalty  float64  `json:"frequency_penalty"`
    PresencePenalty   float64  `json:"presence_penalty"`
    Stop              []string `json:"stop,omitempty"`
    ChatID            *int64   `json:"chat_id,omitempty"`
}

type CompletionResponse struct {
    ID           string    `json:"id"`
    Content      string    `json:"content"`
    FinishReason string    `json:"finish_reason"`
    Usage        Usage     `json:"usage"`
    ChatID       int64     `json:"chat_id,omitempty"`
}

type Token struct {
    Text    string
    IsFinal bool
}

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type Usage struct {
    PromptTokens     int `json:"prompt_tokens"`
    CompletionTokens int `json:"completion_tokens"`
    TotalTokens      int `json:"total_tokens"`
}
```

### Online Agent Implementation

```go
// internal/agent/online_agent.go
package agent

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "net/http"
)

type OnlineAgent struct {
    client   *http.Client
    endpoint string
    apiKey   string
    model    string
    provider Provider
}

type Provider string

const (
    ProviderOpenAI    Provider = "openai"
    ProviderAnthropic Provider = "anthropic"
    ProviderGroq      Provider = "groq"
    ProviderXAI       Provider = "xai"
)

func NewOnlineAgent(provider Provider, model, apiKey string) *OnlineAgent {
    endpoints := map[Provider]string{
        ProviderOpenAI:    "https://api.openai.com/v1/chat/completions",
        ProviderAnthropic: "https://api.anthropic.com/v1/messages",
        ProviderGroq:      "https://api.groq.com/openai/v1/chat/completions",
        ProviderXAI:       "https://api.x.ai/v1/chat/completions",
    }

    return &OnlineAgent{
        client:   &http.Client{Timeout: 120 * time.Second},
        endpoint: endpoints[provider],
        apiKey:   apiKey,
        model:    model,
        provider: provider,
    }
}

func (a *OnlineAgent) CompleteText(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
    // Build provider-specific request
    body, err := a.buildRequestBody(req)
    if err != nil {
        return nil, fmt.Errorf("building request: %w", err)
    }

    httpReq, err := http.NewRequestWithContext(ctx, "POST", a.endpoint, bytes.NewReader(body))
    if err != nil {
        return nil, fmt.Errorf("creating request: %w", err)
    }

    a.setHeaders(httpReq)

    resp, err := a.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("executing request: %w", err)
    }
    defer resp.Body.Close()

    return a.parseResponse(resp)
}
```

### Local Agent (gRPC Client)

```go
// internal/agent/local_agent.go
package agent

import (
    "context"

    pb "geist/internal/inference/proto"
    "google.golang.org/grpc"
)

type LocalAgent struct {
    client pb.InferenceServiceClient
    model  string
}

func NewLocalAgent(conn *grpc.ClientConn, model string) *LocalAgent {
    return &LocalAgent{
        client: pb.NewInferenceServiceClient(conn),
        model:  model,
    }
}

func (a *LocalAgent) CompleteText(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
    grpcReq := &pb.CompletionRequest{
        ModelId:      a.model,
        SystemPrompt: req.SystemPrompt,
        UserPrompt:   req.Prompt,
        Config: &pb.GenerationConfig{
            MaxTokens:        int32(req.MaxTokens),
            Temperature:      float32(req.Temperature),
            TopP:             float32(req.TopP),
            FrequencyPenalty: float32(req.FrequencyPenalty),
            PresencePenalty:  float32(req.PresencePenalty),
            StopSequences:    req.Stop,
        },
    }

    resp, err := a.client.Complete(ctx, grpcReq)
    if err != nil {
        return nil, err
    }

    return &CompletionResponse{
        ID:           resp.Id,
        Content:      resp.Content,
        FinishReason: resp.FinishReason,
        Usage: Usage{
            PromptTokens:     int(resp.Usage.PromptTokens),
            CompletionTokens: int(resp.Usage.CompletionTokens),
            TotalTokens:      int(resp.Usage.TotalTokens),
        },
    }, nil
}

func (a *LocalAgent) StreamCompleteText(ctx context.Context, req *CompletionRequest) (<-chan Token, error) {
    grpcReq := &pb.CompletionRequest{
        ModelId:      a.model,
        SystemPrompt: req.SystemPrompt,
        UserPrompt:   req.Prompt,
        Config: &pb.GenerationConfig{
            MaxTokens:   int32(req.MaxTokens),
            Temperature: float32(req.Temperature),
        },
    }

    stream, err := a.client.StreamComplete(ctx, grpcReq)
    if err != nil {
        return nil, err
    }

    tokens := make(chan Token, 100)
    go func() {
        defer close(tokens)
        for {
            chunk, err := stream.Recv()
            if err != nil {
                return
            }
            tokens <- Token{
                Text:    chunk.Token,
                IsFinal: chunk.IsFinal,
            }
            if chunk.IsFinal {
                return
            }
        }
    }()

    return tokens, nil
}
```

---

## 6. Python MLX Service

### gRPC Server Implementation

```python
# mlx_service/server.py
import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc

from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.architectures.registry import get_runner

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.runners = {}
        self.default_runner = None

    def LoadModel(self, request, context):
        """Load a model into memory."""
        runner_class = get_runner(request.runner_type or "mlx_llama")
        runner = runner_class()
        runner.load(request.model_id, request.device_config)
        self.runners[request.model_id] = runner
        return inference_pb2.LoadModelResponse(success=True)

    def Complete(self, request, context):
        """Synchronous completion."""
        runner = self._get_runner(request.model_id)

        result = runner.complete(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            generation_config={
                'max_tokens': request.config.max_tokens,
                'temperature': request.config.temperature,
                'top_p': request.config.top_p,
                'frequency_penalty': request.config.frequency_penalty,
                'presence_penalty': request.config.presence_penalty,
                'stop': list(request.config.stop_sequences),
            }
        )

        return inference_pb2.CompletionResponse(
            id=result.get('id', ''),
            content=result['content'],
            finish_reason=result.get('finish_reason', 'stop'),
            usage=inference_pb2.UsageStats(
                prompt_tokens=result['usage']['prompt_tokens'],
                completion_tokens=result['usage']['completion_tokens'],
                total_tokens=result['usage']['total_tokens'],
            )
        )

    def StreamComplete(self, request, context):
        """Streaming token generation."""
        runner = self._get_runner(request.model_id)

        for token in runner.stream_generate(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            generation_config={
                'max_tokens': request.config.max_tokens,
                'temperature': request.config.temperature,
            }
        ):
            yield inference_pb2.TokenChunk(
                token=token['text'],
                is_final=token.get('is_final', False),
                finish_reason=token.get('finish_reason', ''),
            )

    def _get_runner(self, model_id):
        if model_id in self.runners:
            return self.runners[model_id]
        if self.default_runner:
            return self.default_runner
        raise grpc.RpcError(f"Model {model_id} not loaded")

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"MLX Inference Service running on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

---

## 7. Database Migration

### Option A: GORM (ORM approach)

```go
// internal/database/models.go
package database

import (
    "time"
    "gorm.io/gorm"
)

type ChatSession struct {
    ID        int64          `gorm:"primaryKey"`
    UserID    int64          `gorm:"index"`
    AgentType string         `gorm:"type:varchar(50)"`
    CreatedAt time.Time
    UpdatedAt time.Time
    Messages  []ChatMessage  `gorm:"foreignKey:SessionID"`
}

type ChatMessage struct {
    ID        int64  `gorm:"primaryKey"`
    SessionID int64  `gorm:"index"`
    Role      string `gorm:"type:varchar(20)"`
    Content   string `gorm:"type:text"`
    CreatedAt time.Time
}

type AgentPreset struct {
    ID           int64  `gorm:"primaryKey"`
    Name         string `gorm:"type:varchar(100);uniqueIndex"`
    AgentType    string `gorm:"type:varchar(50)"`
    Model        string `gorm:"type:varchar(100)"`
    SystemPrompt string `gorm:"type:text"`
    Config       string `gorm:"type:jsonb"` // JSON configuration
    CreatedAt    time.Time
    UpdatedAt    time.Time
}

type UserSettings struct {
    ID                   int64  `gorm:"primaryKey"`
    UserID               int64  `gorm:"uniqueIndex"`
    DefaultAgentType     string `gorm:"type:varchar(50)"`
    DefaultModel         string `gorm:"type:varchar(100)"`
    DefaultSystemPrompt  string `gorm:"type:text"`
    Preferences          string `gorm:"type:jsonb"`
    CreatedAt            time.Time
    UpdatedAt            time.Time
}
```

### Option B: sqlc (Generated from SQL)

```sql
-- database/queries/chat_session.sql

-- name: CreateChatSession :one
INSERT INTO chat_sessions (user_id, agent_type, created_at, updated_at)
VALUES ($1, $2, NOW(), NOW())
RETURNING *;

-- name: GetChatSession :one
SELECT * FROM chat_sessions WHERE id = $1;

-- name: AddChatMessage :one
INSERT INTO chat_messages (session_id, role, content, created_at)
VALUES ($1, $2, $3, NOW())
RETURNING *;

-- name: GetChatHistory :many
SELECT * FROM chat_messages
WHERE session_id = $1
ORDER BY created_at ASC;
```

### Recommendation: sqlc

sqlc generates type-safe Go code from SQL queries, offering:
- Better control over exact queries
- No runtime reflection overhead
- Compile-time query validation
- Compatible with existing PostgreSQL schema

---

## 8. Configuration Management

```go
// internal/config/config.go
package config

import (
    "github.com/kelseyhightower/envconfig"
)

type Config struct {
    // Server
    ServerHost string `envconfig:"SERVER_HOST" default:"0.0.0.0"`
    ServerPort int    `envconfig:"SERVER_PORT" default:"8000"`

    // Database
    DatabaseURL string `envconfig:"DATABASE_URL" required:"true"`

    // MLX Service
    MLXServiceAddr string `envconfig:"MLX_SERVICE_ADDR" default:"localhost:50051"`

    // API Keys
    OpenAIKey    string `envconfig:"OPENAI_API_KEY"`
    AnthropicKey string `envconfig:"ANTHROPIC_API_KEY"`
    GroqKey      string `envconfig:"GROQ_API_KEY"`
    XAIKey       string `envconfig:"GROK_API_KEY"`

    // Model Configuration
    LocalWeightsDir string `envconfig:"LOCAL_WEIGHTS_DIR" default:"./weights"`
    DefaultModel    string `envconfig:"DEFAULT_MODEL" default:"meta-llama/Llama-3.1-8B-Instruct"`

    // Logging
    LogLevel        string `envconfig:"LOG_LEVEL" default:"info"`
    EnhancedLogging bool   `envconfig:"ENHANCED_LOGGING" default:"false"`
}

func Load() (*Config, error) {
    var cfg Config
    if err := envconfig.Process("", &cfg); err != nil {
        return nil, err
    }
    return &cfg, nil
}
```

---

## 9. HTTP Handler Example

```go
// internal/api/handlers/agent.go
package handlers

import (
    "encoding/json"
    "net/http"

    "geist/internal/agent"
    "geist/internal/models"
)

type AgentHandler struct {
    factory *agent.Factory
}

func NewAgentHandler(factory *agent.Factory) *AgentHandler {
    return &AgentHandler{factory: factory}
}

func (h *AgentHandler) CompleteText(w http.ResponseWriter, r *http.Request) {
    var req models.CompleteTextRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // Get or create agent
    ag, err := h.factory.GetAgent(req.AgentType)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    // Execute completion
    resp, err := ag.CompleteText(r.Context(), &agent.CompletionRequest{
        Prompt:           req.Prompt,
        SystemPrompt:     req.SystemPrompt,
        MaxTokens:        req.MaxTokens,
        Temperature:      req.Temperature,
        TopP:             req.TopP,
        FrequencyPenalty: req.FrequencyPenalty,
        PresencePenalty:  req.PresencePenalty,
        Stop:             req.Stop,
    })
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

func (h *AgentHandler) StreamCompleteText(w http.ResponseWriter, r *http.Request) {
    var req models.CompleteTextRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    ag, err := h.factory.GetAgent(req.AgentType)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    // Set up SSE
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "Streaming not supported", http.StatusInternalServerError)
        return
    }

    tokens, err := ag.StreamCompleteText(r.Context(), &agent.CompletionRequest{
        Prompt:      req.Prompt,
        MaxTokens:   req.MaxTokens,
        Temperature: req.Temperature,
    })
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    for token := range tokens {
        data, _ := json.Marshal(token)
        fmt.Fprintf(w, "data: %s\n\n", data)
        flusher.Flush()
    }
}
```

---

## 10. Migration Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Go project structure
- [ ] Define protobuf interface for MLX service
- [ ] Create Python gRPC server wrapper around existing MLX code
- [ ] Implement basic Go gRPC client
- [ ] Verify inference round-trip works

### Phase 2: Core Server (Weeks 3-4)
- [ ] Implement HTTP router with Chi
- [ ] Port request/response models to Go structs
- [ ] Implement OnlineAgent (OpenAI, Anthropic, Groq)
- [ ] Implement LocalAgent (gRPC to MLX)
- [ ] Create agent factory

### Phase 3: Database Layer (Week 5)
- [ ] Set up sqlc or GORM
- [ ] Migrate existing SQLAlchemy models
- [ ] Implement repositories for chat sessions, presets, settings
- [ ] Verify compatibility with existing PostgreSQL data

### Phase 4: API Parity (Weeks 6-7)
- [ ] Port all `/agent/*` endpoints
- [ ] Port `/api/v1/models` endpoints
- [ ] Port `/api/v1/voice` endpoints
- [ ] Port `/api/v1/workflows` endpoints
- [ ] Port `/api/v1/files` endpoints
- [ ] Port `/api/v1/user-settings` endpoints

### Phase 5: Adapters (Week 8)
- [ ] Port adapter registry pattern
- [ ] Implement SendGrid adapter
- [ ] Implement search adapter
- [ ] Implement MMS speech-to-text adapter

### Phase 6: Testing & Integration (Weeks 9-10)
- [ ] Write unit tests for all components
- [ ] Write integration tests
- [ ] Set up CI/CD pipeline
- [ ] Performance benchmarking vs Python version
- [ ] Load testing

### Phase 7: Deployment (Week 11)
- [ ] Update Dockerfile for Go binary + Python MLX sidecar
- [ ] Update docker-compose.yml
- [ ] Blue-green deployment strategy
- [ ] Monitoring and alerting setup

### Phase 8: Cleanup (Week 12)
- [ ] Remove deprecated Python server code
- [ ] Update documentation
- [ ] Performance optimization based on production metrics

---

## 11. Performance Expectations

| Metric | Python (FastAPI) | Go (Expected) |
|--------|------------------|---------------|
| Cold start | ~3-5s | <500ms |
| Memory (idle) | ~200-400MB | ~20-50MB |
| Request latency (no inference) | ~5-10ms | ~1-2ms |
| Concurrent connections | ~1000 | ~10,000+ |
| Binary size | N/A (interpreter) | ~15-30MB |

**Note:** Inference latency will remain unchanged as it's handled by the Python MLX subprocess.

---

## 12. Risk Mitigation

### Risk 1: MLX Interop Complexity
**Mitigation:** Start with the simplest working solution (HTTP JSON-RPC), then optimize to gRPC if needed. Keep Python MLX code unchanged.

### Risk 2: API Compatibility Breaking
**Mitigation:** Implement comprehensive API contract tests. Run both Python and Go servers in parallel during migration, compare responses.

### Risk 3: Database Migration Issues
**Mitigation:** Use read-only database access initially. Run migrations only after thorough testing in staging.

### Risk 4: Performance Regression
**Mitigation:** Establish baseline benchmarks before migration. Continuous performance testing throughout development.

---

## 13. Success Criteria

1. **Functional Parity:** All existing API endpoints work identically
2. **Performance:** Non-inference request latency reduced by 50%+
3. **Resource Usage:** Memory footprint reduced by 70%+
4. **Reliability:** Zero regression in error rates
5. **Maintainability:** Comprehensive test coverage (>80%)
6. **Deployment:** Single binary + Python sidecar deployment model

---

## 14. Future Considerations

### Post-Migration Optimizations

1. **Replace Python MLX with Rust/C++:** Once stable, consider rewriting the inference service in Rust using `mlx-rs` bindings for even lower latency.

2. **WebSocket Support:** Native Go WebSocket support for real-time streaming.

3. **Kubernetes Deployment:** Separate Go server and MLX inference pods for independent scaling.

4. **Model Caching:** Implement shared model weights across inference instances using memory-mapped files.

5. **Metrics & Tracing:** Integrate OpenTelemetry for distributed tracing across Go server and Python inference.

---

## Appendix A: Key Go Dependencies

```go
// go.mod
module geist

go 1.22

require (
    github.com/go-chi/chi/v5 v5.0.12      // HTTP router
    google.golang.org/grpc v1.62.0        // gRPC
    google.golang.org/protobuf v1.33.0    // Protocol buffers
    github.com/jackc/pgx/v5 v5.5.4        // PostgreSQL driver
    github.com/kelseyhightower/envconfig v1.4.0  // Config
    github.com/rs/zerolog v1.32.0         // Logging
    github.com/stretchr/testify v1.9.0    // Testing
)
```

## Appendix B: Docker Deployment

```dockerfile
# Dockerfile
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /geist-server ./cmd/server

# Final image
FROM python:3.11-slim

# Install Python dependencies for MLX service
COPY mlx_service/requirements.txt /mlx_service/
RUN pip install --no-cache-dir -r /mlx_service/requirements.txt

# Copy Go binary
COPY --from=builder /geist-server /usr/local/bin/

# Copy MLX service
COPY mlx_service/ /mlx_service/
COPY agents/ /agents/

# Entrypoint script to start both services
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000 50051

ENTRYPOINT ["/docker-entrypoint.sh"]
```

```bash
#!/bin/bash
# docker-entrypoint.sh

# Start MLX inference service in background
python /mlx_service/server.py &

# Wait for MLX service to be ready
sleep 2

# Start Go server
exec /usr/local/bin/geist-server
```
