# Rust Migration Plan for Geist Core

## Executive Summary

This document outlines a phased approach to migrating Geist's core server components to Rust while maintaining MLX interoperability for inference on Apple Silicon. The goal is to achieve better performance, memory safety, and concurrency without sacrificing the existing MLX inference capabilities.

## Current Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                              │
│                         (app/main.py)                               │
├─────────────────────────────────────────────────────────────────────┤
│                          Agent Layer                                │
│   ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    │
│   │ LocalAgent   │    │ OnlineAgent   │    │ LlamaAgent       │    │
│   └──────────────┘    └───────────────┘    └──────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                         Runner Layer                                │
│   ┌──────────────────┐    ┌─────────────┐    ┌────────────────┐    │
│   │ MLXLlamaRunner   │    │ VLLMRunner  │    │ Future Runners │    │
│   └──────────────────┘    └─────────────┘    └────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                      Inference Layer (MLX)                          │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                       LlamaMLX                               │  │
│   │   (mlx.core, mlx.nn, Attention, TransformerBlock, etc.)     │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Rust HTTP Server (Axum/Actix)                    │
│                         + Tower middleware                          │
├─────────────────────────────────────────────────────────────────────┤
│                     Rust Agent Orchestration                        │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │   AgentManager, GenerationConfig, RequestRouter               │ │
│   └──────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    Rust Runner Interface Layer                      │
│   ┌──────────────────┐    ┌──────────────────────────────────────┐ │
│   │  HTTP Runners    │    │    MLX Interop (PyO3 Bridge)         │ │
│   │  (Pure Rust)     │    │    ↓ calls Python                    │ │
│   └──────────────────┘    └──────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                      Python Inference Layer                         │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │           LlamaMLX (unchanged MLX implementation)            │  │
│   │           Exposed as Python module for Rust FFI              │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation & Project Setup

### 1.1 Create Rust Workspace Structure

```
geist-core/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── geist-server/             # Main HTTP server
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── routes/
│   │       ├── middleware/
│   │       └── config.rs
│   │
│   ├── geist-agents/             # Agent orchestration
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── agent.rs
│   │       ├── context.rs
│   │       └── factory.rs
│   │
│   ├── geist-runners/            # Runner implementations
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs         # Runner trait definitions
│   │       ├── http_runner.rs    # Online inference
│   │       └── mlx_bridge.rs     # PyO3 bridge to MLX
│   │
│   ├── geist-types/              # Shared types
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── generation.rs     # GenerationConfig, etc.
│   │       ├── completion.rs     # Completion responses
│   │       └── errors.rs
│   │
│   └── geist-python/             # PyO3 module for Python interop
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
│
└── python/
    └── geist_mlx/                # Python package for MLX inference
        ├── __init__.py
        ├── runner.py             # Thin wrapper exposing LlamaMLX
        └── types.py              # Pydantic models matching Rust
```

### 1.2 Core Dependencies

**Cargo.toml (workspace)**:
```toml
[workspace]
members = [
    "crates/geist-server",
    "crates/geist-agents",
    "crates/geist-runners",
    "crates/geist-types",
    "crates/geist-python",
]
resolver = "2"

[workspace.dependencies]
# Web framework
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
tokio = { version = "1", features = ["full"] }
hyper = "1.0"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Python interop
pyo3 = { version = "0.21", features = ["auto-initialize", "gil-refs"] }

# Database
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres", "uuid", "chrono"] }

# HTTP client (for online providers)
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }

# Utilities
thiserror = "1"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
async-trait = "0.1"
```

---

## Phase 2: Core Types & Traits (Rust)

### 2.1 Generation Configuration

**crates/geist-types/src/generation.rs**:
```rust
use serde::{Deserialize, Serialize};

/// Configuration for text generation - mirrors Python GenerationConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,

    /// Stop sequences that halt generation
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

fn default_timeout() -> u64 { 120 }

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: vec![],
            timeout_seconds: 120,
        }
    }
}
```

### 2.2 Runner Trait

**crates/geist-runners/src/traits.rs**:
```rust
use async_trait::async_trait;
use geist_types::{GenerationConfig, Completion, GeistError};

/// Core abstraction for inference backends
#[async_trait]
pub trait Runner: Send + Sync {
    /// Run text completion
    async fn complete(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Completion, GeistError>;

    /// Check if the runner is available/healthy
    async fn health_check(&self) -> Result<(), GeistError>;

    /// Get runner metadata
    fn metadata(&self) -> RunnerMetadata;
}

#[derive(Debug, Clone)]
pub struct RunnerMetadata {
    pub name: String,
    pub model_id: String,
    pub backend: Backend,
    pub supports_streaming: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    MLX,        // Local Apple Silicon via Python interop
    VLLM,       // vLLM server
    OpenAI,     // OpenAI API
    Anthropic,  // Anthropic API
    Groq,       // Groq API
    Custom,     // Custom HTTP endpoint
}
```

---

## Phase 3: MLX Python Interop

### 3.1 Strategy: PyO3 Bridge

The MLX library is Python/C++ only with no Rust bindings. We'll use PyO3 to call into Python:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Rust Application                          │
│                                                                  │
│  ┌──────────────┐     ┌────────────────┐     ┌───────────────┐  │
│  │ HTTP Request │ ──► │ Rust Agent     │ ──► │ MLXBridge     │  │
│  └──────────────┘     └────────────────┘     └───────┬───────┘  │
│                                                       │          │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─   │
│                                                       │ PyO3     │
│                                                       ▼ FFI      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Python GIL Context                      │   │
│  │                                                            │   │
│  │  ┌─────────────────┐     ┌───────────────────────────┐    │   │
│  │  │ geist_mlx.runner│ ──► │ LlamaMLX (existing code) │    │   │
│  │  └─────────────────┘     └───────────────────────────┘    │   │
│  │                                                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Python Wrapper Module

**python/geist_mlx/runner.py**:
```python
"""
Thin Python wrapper exposing LlamaMLX to Rust via PyO3.
This module provides a stable interface for the Rust FFI bridge.
"""
from dataclasses import dataclass
from typing import Optional, List
import json

# Import existing MLX implementation
from agents.architectures.llama.llama_mlx import LlamaMLX

# Singleton model cache (avoid reloading on each call)
_model_cache: dict[str, LlamaMLX] = {}


@dataclass
class GenerationConfig:
    """Matches Rust GenerationConfig struct"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    timeout_seconds: int = 120

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationConfig":
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class CompletionResult:
    """Result from MLX inference - serializable for Rust"""
    text: str
    tokens_generated: int
    finish_reason: str  # "stop", "length", "timeout"

    def to_json(self) -> str:
        return json.dumps({
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "finish_reason": self.finish_reason,
        })


def get_or_create_model(model_id: str) -> LlamaMLX:
    """Get cached model or create new one."""
    if model_id not in _model_cache:
        _model_cache[model_id] = LlamaMLX(model=model_id)
    return _model_cache[model_id]


def complete(
    model_id: str,
    system_prompt: Optional[str],
    user_prompt: str,
    config_json: str,
) -> str:
    """
    Main entry point for Rust FFI.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        system_prompt: Optional system prompt
        user_prompt: User's input text
        config_json: JSON-serialized GenerationConfig

    Returns:
        JSON-serialized CompletionResult
    """
    config = GenerationConfig.from_json(config_json)
    model = get_or_create_model(model_id)

    # Call existing LlamaMLX implementation
    result = model.complete(
        system_prompt=system_prompt or "",
        user_prompt=user_prompt,
        max_tokens=config.max_tokens,
        temp=config.temperature,
    )

    # Convert to serializable result
    completion = CompletionResult(
        text=result.llama_completion,
        tokens_generated=result.tokens,
        finish_reason="stop",  # TODO: Detect actual finish reason
    )

    return completion.to_json()


def health_check() -> bool:
    """Verify MLX is available."""
    try:
        import mlx.core as mx
        # Simple operation to verify MLX works
        _ = mx.array([1.0, 2.0, 3.0])
        return True
    except Exception:
        return False
```

### 3.3 Rust PyO3 Bridge

**crates/geist-runners/src/mlx_bridge.rs**:
```rust
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::OnceLock;
use tokio::sync::Mutex;

use geist_types::{GenerationConfig, Completion, GeistError};
use crate::traits::{Runner, RunnerMetadata, Backend};

/// Singleton to hold the Python interpreter
static PYTHON_RUNTIME: OnceLock<Mutex<PythonRuntime>> = OnceLock::new();

struct PythonRuntime {
    initialized: bool,
}

impl PythonRuntime {
    fn new() -> Self {
        // Initialize Python interpreter
        pyo3::prepare_freethreaded_python();
        Self { initialized: true }
    }
}

/// MLX inference runner that bridges to Python
pub struct MLXRunner {
    model_id: String,
}

impl MLXRunner {
    pub fn new(model_id: impl Into<String>) -> Self {
        // Ensure Python runtime is initialized
        PYTHON_RUNTIME.get_or_init(|| Mutex::new(PythonRuntime::new()));

        Self {
            model_id: model_id.into(),
        }
    }

    fn call_python_complete(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Completion, GeistError> {
        Python::with_gil(|py| {
            // Import our wrapper module
            let geist_mlx = PyModule::import(py, "geist_mlx.runner")
                .map_err(|e| GeistError::PythonError(e.to_string()))?;

            // Serialize config to JSON
            let config_json = serde_json::to_string(config)
                .map_err(|e| GeistError::SerializationError(e.to_string()))?;

            // Call the complete function
            let result: String = geist_mlx
                .getattr("complete")
                .map_err(|e| GeistError::PythonError(e.to_string()))?
                .call1((
                    &self.model_id,
                    system_prompt,
                    user_prompt,
                    config_json,
                ))
                .map_err(|e| GeistError::InferenceError(e.to_string()))?
                .extract()
                .map_err(|e| GeistError::PythonError(e.to_string()))?;

            // Deserialize result
            let completion: Completion = serde_json::from_str(&result)
                .map_err(|e| GeistError::SerializationError(e.to_string()))?;

            Ok(completion)
        })
    }
}

#[async_trait::async_trait]
impl Runner for MLXRunner {
    async fn complete(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Completion, GeistError> {
        // Run Python call in blocking task to not block tokio runtime
        let model_id = self.model_id.clone();
        let system = system_prompt.map(String::from);
        let user = user_prompt.to_string();
        let cfg = config.clone();

        tokio::task::spawn_blocking(move || {
            let runner = MLXRunner { model_id };
            runner.call_python_complete(system.as_deref(), &user, &cfg)
        })
        .await
        .map_err(|e| GeistError::RuntimeError(e.to_string()))?
    }

    async fn health_check(&self) -> Result<(), GeistError> {
        tokio::task::spawn_blocking(|| {
            Python::with_gil(|py| {
                let geist_mlx = PyModule::import(py, "geist_mlx.runner")?;
                let healthy: bool = geist_mlx.getattr("health_check")?.call0()?.extract()?;
                if healthy {
                    Ok(())
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("MLX not available"))
                }
            })
        })
        .await
        .map_err(|e| GeistError::RuntimeError(e.to_string()))?
        .map_err(|e: PyErr| GeistError::PythonError(e.to_string()))
    }

    fn metadata(&self) -> RunnerMetadata {
        RunnerMetadata {
            name: "MLX".into(),
            model_id: self.model_id.clone(),
            backend: Backend::MLX,
            supports_streaming: false, // TODO: Implement streaming
        }
    }
}
```

---

## Phase 4: HTTP Server (Rust)

### 4.1 Axum Server Implementation

**crates/geist-server/src/main.rs**:
```rust
use axum::{
    Router,
    routing::{get, post},
    extract::State,
    Json,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing_subscriber;

mod routes;
mod config;
mod state;

use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load configuration
    let config = config::Config::from_env()?;

    // Initialize application state
    let state = Arc::new(AppState::new(&config).await?);

    // Build router
    let app = Router::new()
        .route("/health", get(routes::health_check))
        .route("/agent/complete_text", post(routes::complete_text))
        .route("/agent/complete_text_new", post(routes::complete_text_new))
        .nest("/api/v1", routes::api_v1_router())
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting Geist server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

### 4.2 Route Handlers

**crates/geist-server/src/routes/agent.rs**:
```rust
use axum::{
    extract::{State, Path},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

use geist_types::{GenerationConfig, Completion};
use geist_agents::{AgentFactory, AgentType};
use crate::state::AppState;
use crate::error::ApiError;

#[derive(serde::Deserialize)]
pub struct CompleteTextRequest {
    pub agent_type: AgentType,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub generation_config: Option<GenerationConfig>,
}

#[derive(serde::Serialize)]
pub struct CompleteTextResponse {
    pub completion: Completion,
    pub session_id: Option<Uuid>,
}

pub async fn complete_text_new(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompleteTextRequest>,
) -> Result<Json<CompleteTextResponse>, ApiError> {
    // Get or create agent
    let agent = state.agent_cache
        .get_or_create(request.agent_type, request.model.as_deref())
        .await?;

    // Run completion
    let config = request.generation_config.unwrap_or_default();
    let completion = agent
        .complete_text(
            request.system_prompt.as_deref(),
            &request.user_prompt,
            &config,
        )
        .await?;

    // TODO: Persist to chat history

    Ok(Json(CompleteTextResponse {
        completion,
        session_id: None,
    }))
}
```

---

## Phase 5: Agent Orchestration (Rust)

### 5.1 Agent Trait & Implementations

**crates/geist-agents/src/agent.rs**:
```rust
use async_trait::async_trait;
use geist_types::{GenerationConfig, Completion, GeistError};
use geist_runners::traits::Runner;
use std::sync::Arc;

/// Core agent trait
#[async_trait]
pub trait Agent: Send + Sync {
    /// Complete text generation
    async fn complete_text(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Completion, GeistError>;

    /// Get agent type
    fn agent_type(&self) -> AgentType;
}

/// Agent for local inference (MLX)
pub struct LocalAgent {
    runner: Arc<dyn Runner>,
    default_system_prompt: Option<String>,
}

impl LocalAgent {
    pub fn new(runner: Arc<dyn Runner>) -> Self {
        Self {
            runner,
            default_system_prompt: None,
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.default_system_prompt = Some(prompt.into());
        self
    }
}

#[async_trait]
impl Agent for LocalAgent {
    async fn complete_text(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Completion, GeistError> {
        let system = system_prompt.or(self.default_system_prompt.as_deref());
        self.runner.complete(system, user_prompt, config).await
    }

    fn agent_type(&self) -> AgentType {
        AgentType::Local
    }
}

/// Agent for HTTP-based providers (OpenAI, Anthropic, etc.)
pub struct OnlineAgent {
    runner: Arc<dyn Runner>,
    provider: Provider,
}

#[derive(Debug, Clone, Copy)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Groq,
    Custom,
}

#[async_trait]
impl Agent for OnlineAgent {
    async fn complete_text(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Completion, GeistError> {
        self.runner.complete(system_prompt, user_prompt, config).await
    }

    fn agent_type(&self) -> AgentType {
        AgentType::Online
    }
}
```

### 5.2 Agent Factory

**crates/geist-agents/src/factory.rs**:
```rust
use std::sync::Arc;
use geist_runners::{MLXRunner, HttpRunner};
use crate::agent::{Agent, LocalAgent, OnlineAgent, Provider};
use geist_types::GeistError;

pub struct AgentFactory;

impl AgentFactory {
    /// Create a LocalAgent with MLX runner
    pub fn create_local(model_id: &str) -> Result<Arc<dyn Agent>, GeistError> {
        let runner = Arc::new(MLXRunner::new(model_id));
        let agent = LocalAgent::new(runner);
        Ok(Arc::new(agent))
    }

    /// Create an OnlineAgent for HTTP providers
    pub fn create_online(
        provider: Provider,
        api_key: &str,
        model: &str,
    ) -> Result<Arc<dyn Agent>, GeistError> {
        let (base_url, model_param) = match provider {
            Provider::OpenAI => ("https://api.openai.com/v1", model),
            Provider::Anthropic => ("https://api.anthropic.com/v1", model),
            Provider::Groq => ("https://api.groq.com/openai/v1", model),
            Provider::Custom => return Err(GeistError::ConfigError(
                "Custom provider requires explicit URL".into()
            )),
        };

        let runner = Arc::new(HttpRunner::new(base_url, api_key, model_param));
        let agent = OnlineAgent { runner, provider };
        Ok(Arc::new(agent))
    }
}
```

---

## Phase 6: Database Integration

### 6.1 SQLx Migrations

The existing PostgreSQL schema can be preserved. SQLx will handle database access:

**crates/geist-server/src/db/mod.rs**:
```rust
use sqlx::postgres::{PgPool, PgPoolOptions};
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
}

#[derive(sqlx::FromRow)]
pub struct ChatSession {
    pub id: Uuid,
    pub user_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub title: Option<String>,
}

#[derive(sqlx::FromRow)]
pub struct ChatMessage {
    pub id: Uuid,
    pub session_id: Uuid,
    pub role: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

impl ChatSession {
    pub async fn get_messages(
        pool: &PgPool,
        session_id: Uuid,
    ) -> Result<Vec<ChatMessage>, sqlx::Error> {
        sqlx::query_as!(
            ChatMessage,
            r#"
            SELECT id, session_id, role, content, created_at
            FROM chat_messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            "#,
            session_id
        )
        .fetch_all(pool)
        .await
    }
}
```

---

## Phase 7: Migration Strategy

### 7.1 Parallel Running (Hybrid Mode)

During migration, run both servers simultaneously:

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │   (nginx/caddy) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────────┐ ┌────────────────┐
     │ Python FastAPI │ │ Rust Axum      │
     │ (existing)     │ │ (new)          │
     │ Port 5001      │ │ Port 5002      │
     └────────────────┘ └────────────────┘
              │              │
              │              │
              ▼              ▼
     ┌─────────────────────────────────┐
     │        PostgreSQL + Redis       │
     │          (shared state)         │
     └─────────────────────────────────┘
```

### 7.2 Feature Flags for Gradual Rollout

```rust
// config.rs
pub struct FeatureFlags {
    /// Use Rust server for /agent/complete_text_new
    pub rust_complete_text: bool,

    /// Use Rust server for chat history
    pub rust_chat_history: bool,

    /// Use Rust server for all agent routes
    pub rust_agent_routes: bool,
}
```

### 7.3 Migration Order

| Phase | Components | Risk | Duration |
|-------|-----------|------|----------|
| 1 | Types, traits, project structure | Low | 1-2 weeks |
| 2 | MLX Python bridge (PyO3) | Medium | 2-3 weeks |
| 3 | HTTP runners (OpenAI, Anthropic) | Low | 1-2 weeks |
| 4 | Agent orchestration layer | Medium | 2-3 weeks |
| 5 | HTTP server + routes | Medium | 2-3 weeks |
| 6 | Database layer (SQLx) | Low | 1-2 weeks |
| 7 | Full integration + testing | High | 2-4 weeks |
| 8 | Hybrid mode deployment | Medium | 1-2 weeks |
| 9 | Complete cutover | High | 1 week |

**Total Estimated: 13-22 weeks**

---

## Phase 8: Performance Optimizations

### 8.1 Async Python Interop

For better concurrency, use `pyo3-asyncio` for async Python calls:

```rust
use pyo3_asyncio::tokio::future_into_py;

async fn call_mlx_async(
    py: Python<'_>,
    model_id: &str,
    prompt: &str,
) -> PyResult<String> {
    // Run Python coroutine from Rust async context
    let coro = py.import("geist_mlx.async_runner")?
        .getattr("complete_async")?
        .call1((model_id, prompt))?;

    pyo3_asyncio::tokio::into_future(coro).await
}
```

### 8.2 Model Caching Strategy

```rust
use dashmap::DashMap;
use std::sync::Arc;

/// Thread-safe model cache
pub struct ModelCache {
    models: DashMap<String, Arc<dyn Runner>>,
    max_models: usize,
}

impl ModelCache {
    pub fn new(max_models: usize) -> Self {
        Self {
            models: DashMap::new(),
            max_models,
        }
    }

    pub async fn get_or_load(&self, model_id: &str) -> Arc<dyn Runner> {
        // Check cache first
        if let Some(runner) = self.models.get(model_id) {
            return runner.clone();
        }

        // Load model (this is the expensive operation)
        let runner = Arc::new(MLXRunner::new(model_id));

        // Evict if at capacity (LRU would be better)
        if self.models.len() >= self.max_models {
            if let Some(key) = self.models.iter().next().map(|e| e.key().clone()) {
                self.models.remove(&key);
            }
        }

        self.models.insert(model_id.to_string(), runner.clone());
        runner
    }
}
```

### 8.3 Request Batching (Future)

For throughput optimization, batch multiple requests:

```rust
pub struct BatchingRunner {
    inner: Arc<dyn Runner>,
    batch_size: usize,
    batch_timeout: Duration,
    pending: Mutex<Vec<PendingRequest>>,
}

struct PendingRequest {
    prompt: String,
    config: GenerationConfig,
    response_tx: oneshot::Sender<Completion>,
}
```

---

## Phase 9: Testing Strategy

### 9.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 2048);
        assert!((config.temperature - 0.7).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mlx_runner_health() {
        let runner = MLXRunner::new("meta-llama/Llama-3.1-8B-Instruct");
        // This will fail if MLX isn't available
        let result = runner.health_check().await;
        assert!(result.is_ok());
    }
}
```

### 9.2 Integration Tests

```rust
#[tokio::test]
async fn test_complete_text_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/agent/complete_text_new")
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{
                    "agent_type": "local",
                    "user_prompt": "Hello, world!"
                }"#))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

### 9.3 Python Interop Tests

```rust
#[test]
fn test_python_interop() {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let result: bool = py
            .import("geist_mlx.runner")
            .unwrap()
            .getattr("health_check")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();

        assert!(result);
    });
}
```

---

## Phase 10: Deployment Considerations

### 10.1 Build Configuration

**Dockerfile.rust**:
```dockerfile
# Build stage
FROM rust:1.75-bookworm as builder

WORKDIR /app
COPY . .
RUN cargo build --release

# Runtime stage
FROM python:3.11-slim-bookworm

# Install Rust binary
COPY --from=builder /app/target/release/geist-server /usr/local/bin/

# Install Python dependencies for MLX interop
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python modules
COPY python/ /app/python/
ENV PYTHONPATH=/app/python:$PYTHONPATH

EXPOSE 8000
CMD ["geist-server"]
```

### 10.2 Environment Variables

```bash
# Rust server config
GEIST_HOST=0.0.0.0
GEIST_PORT=8000
GEIST_LOG_LEVEL=info

# Database
DATABASE_URL=postgres://user:pass@localhost/geist

# Python/MLX config
PYTHONPATH=/app/python
MLX_DEFAULT_MODEL=meta-llama/Llama-3.1-8B-Instruct

# API keys (for online providers)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## Key Decisions & Trade-offs

### Why PyO3 for MLX Interop?

| Option | Pros | Cons |
|--------|------|------|
| **PyO3 (chosen)** | Mature, well-documented, maintains Python GIL safety | Requires Python runtime, GIL contention |
| C FFI | No Python dependency | MLX has no C API, would need wrappers |
| gRPC service | Clean separation | Added latency, operational complexity |
| Rewrite MLX in Rust | Pure Rust stack | Enormous effort, no Apple support |

### Why Axum over Actix?

- Better async ecosystem integration (tower middleware)
- Simpler mental model
- Strong type safety with extractors
- Active maintenance and community

### What Stays in Python?

1. **MLX inference** - No Rust bindings, Apple's framework
2. **Model loading** - HuggingFace transformers integration
3. **Tokenization** - Can migrate later with `tokenizers` crate

---

## Success Metrics

| Metric | Current (Python) | Target (Rust) |
|--------|-----------------|---------------|
| Request latency (p50) | ~50ms overhead | <10ms overhead |
| Memory usage (idle) | ~500MB | <100MB |
| Concurrent connections | ~100 | >10,000 |
| Cold start time | ~5s | <1s |
| Binary size | N/A (interpreted) | <50MB |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyO3 GIL contention | Medium | High | Use `spawn_blocking`, batch requests |
| MLX API changes | Low | Medium | Pin MLX version, abstraction layer |
| Database migration issues | Low | High | Use same schema, test migrations |
| Performance regression | Medium | High | Benchmark continuously, A/B test |
| Team Rust experience | Variable | Medium | Training, code reviews, documentation |

---

## Appendix A: File Mapping

| Python File | Rust Equivalent |
|------------|-----------------|
| `app/main.py` | `crates/geist-server/src/main.rs` |
| `agents/base_agent.py` | `crates/geist-agents/src/agent.rs` |
| `agents/local_agent.py` | `crates/geist-agents/src/agent.rs::LocalAgent` |
| `agents/online_agent.py` | `crates/geist-agents/src/agent.rs::OnlineAgent` |
| `agents/factory.py` | `crates/geist-agents/src/factory.rs` |
| `agents/architectures/mlx_llama_runner.py` | `crates/geist-runners/src/mlx_bridge.rs` |
| `agents/models/generation_config.py` | `crates/geist-types/src/generation.rs` |
| `agents/models/completion.py` | `crates/geist-types/src/completion.rs` |
| `app/services/chat_session.py` | `crates/geist-server/src/db/chat.rs` |

## Appendix B: Recommended Reading

1. [PyO3 User Guide](https://pyo3.rs/)
2. [Axum Documentation](https://docs.rs/axum/latest/axum/)
3. [SQLx Guide](https://github.com/launchbadge/sqlx)
4. [MLX Documentation](https://ml-explore.github.io/mlx/)
5. [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
