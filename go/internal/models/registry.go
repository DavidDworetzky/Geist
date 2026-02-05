// Package models provides data models for API requests and responses.
package models

import (
	"fmt"
	"strings"
	"sync"
)

// Provider represents an LLM provider.
type Provider string

const (
	ProviderOpenAI     Provider = "openai"
	ProviderAnthropic  Provider = "anthropic"
	ProviderGroq       Provider = "groq"
	ProviderXAI        Provider = "xai"
	ProviderHuggingFace Provider = "huggingface"
	ProviderLocal      Provider = "local"
)

// ModelInfo contains information about an available model.
type ModelInfo struct {
	ID                      string   `json:"id"`
	Name                    string   `json:"name"`
	Provider                Provider `json:"provider"`
	ContextWindow           int      `json:"context_window,omitempty"`
	MaxOutputTokens         int      `json:"max_output_tokens,omitempty"`
	SupportsVision          bool     `json:"supports_vision"`
	SupportsFunctionCalling bool     `json:"supports_function_calling"`
	SupportsStreaming       bool     `json:"supports_streaming"`
	Recommended             bool     `json:"recommended"`
	Family                  string   `json:"family,omitempty"`
	Endpoint                string   `json:"-"` // Internal use only
}

// ModelRegistry maintains a registry of available models.
type ModelRegistry struct {
	mu     sync.RWMutex
	models map[string]ModelInfo
}

// NewModelRegistry creates a new model registry with default models.
func NewModelRegistry() *ModelRegistry {
	r := &ModelRegistry{
		models: make(map[string]ModelInfo),
	}
	r.registerDefaults()
	return r
}

// Register adds a model to the registry.
func (r *ModelRegistry) Register(model ModelInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models[model.ID] = model
}

// Get retrieves a model by ID.
func (r *ModelRegistry) Get(id string) (ModelInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	model, ok := r.models[id]
	return model, ok
}

// List returns all registered models.
func (r *ModelRegistry) List() []ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	models := make([]ModelInfo, 0, len(r.models))
	for _, m := range r.models {
		models = append(models, m)
	}
	return models
}

// ListByProvider returns models for a specific provider.
func (r *ModelRegistry) ListByProvider(provider Provider) []ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var models []ModelInfo
	for _, m := range r.models {
		if m.Provider == provider {
			models = append(models, m)
		}
	}
	return models
}

// GetEndpoint returns the API endpoint for a provider.
func GetEndpoint(provider Provider) string {
	endpoints := map[Provider]string{
		ProviderOpenAI:    "https://api.openai.com/v1/chat/completions",
		ProviderAnthropic: "https://api.anthropic.com/v1/messages",
		ProviderGroq:      "https://api.groq.com/openai/v1/chat/completions",
		ProviderXAI:       "https://api.x.ai/v1/chat/completions",
	}
	return endpoints[provider]
}

// GetProviderForModel determines the provider for a model ID.
func GetProviderForModel(modelID string) (Provider, error) {
	modelID = strings.ToLower(modelID)

	switch {
	case strings.HasPrefix(modelID, "gpt-") || strings.HasPrefix(modelID, "o1-") || strings.HasPrefix(modelID, "o3-"):
		return ProviderOpenAI, nil
	case strings.HasPrefix(modelID, "claude-"):
		return ProviderAnthropic, nil
	case strings.HasPrefix(modelID, "llama-") || strings.HasPrefix(modelID, "mixtral-") || strings.HasPrefix(modelID, "gemma-"):
		return ProviderGroq, nil
	case strings.HasPrefix(modelID, "grok-"):
		return ProviderXAI, nil
	case strings.Contains(modelID, "/"):
		return ProviderLocal, nil
	default:
		return "", fmt.Errorf("unknown provider for model: %s", modelID)
	}
}

func (r *ModelRegistry) registerDefaults() {
	// OpenAI Models
	r.Register(ModelInfo{
		ID:                      "gpt-4o",
		Name:                    "GPT-4o",
		Provider:                ProviderOpenAI,
		ContextWindow:           128000,
		MaxOutputTokens:         16384,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "gpt-4",
	})
	r.Register(ModelInfo{
		ID:                      "gpt-4o-mini",
		Name:                    "GPT-4o Mini",
		Provider:                ProviderOpenAI,
		ContextWindow:           128000,
		MaxOutputTokens:         16384,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "gpt-4",
	})
	r.Register(ModelInfo{
		ID:                      "gpt-4-turbo",
		Name:                    "GPT-4 Turbo",
		Provider:                ProviderOpenAI,
		ContextWindow:           128000,
		MaxOutputTokens:         4096,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             false,
		Family:                  "gpt-4",
	})
	r.Register(ModelInfo{
		ID:                      "o1-preview",
		Name:                    "O1 Preview",
		Provider:                ProviderOpenAI,
		ContextWindow:           128000,
		MaxOutputTokens:         32768,
		SupportsVision:          false,
		SupportsFunctionCalling: false,
		SupportsStreaming:       false,
		Recommended:             false,
		Family:                  "o1",
	})
	r.Register(ModelInfo{
		ID:                      "o1-mini",
		Name:                    "O1 Mini",
		Provider:                ProviderOpenAI,
		ContextWindow:           128000,
		MaxOutputTokens:         65536,
		SupportsVision:          false,
		SupportsFunctionCalling: false,
		SupportsStreaming:       false,
		Recommended:             false,
		Family:                  "o1",
	})

	// Anthropic Models
	r.Register(ModelInfo{
		ID:                      "claude-3-5-sonnet-20241022",
		Name:                    "Claude 3.5 Sonnet",
		Provider:                ProviderAnthropic,
		ContextWindow:           200000,
		MaxOutputTokens:         8192,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "claude-3.5",
	})
	r.Register(ModelInfo{
		ID:                      "claude-3-5-haiku-20241022",
		Name:                    "Claude 3.5 Haiku",
		Provider:                ProviderAnthropic,
		ContextWindow:           200000,
		MaxOutputTokens:         8192,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "claude-3.5",
	})
	r.Register(ModelInfo{
		ID:                      "claude-3-opus-20240229",
		Name:                    "Claude 3 Opus",
		Provider:                ProviderAnthropic,
		ContextWindow:           200000,
		MaxOutputTokens:         4096,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             false,
		Family:                  "claude-3",
	})

	// Groq Models
	r.Register(ModelInfo{
		ID:                      "llama-3.1-70b-versatile",
		Name:                    "Llama 3.1 70B Versatile",
		Provider:                ProviderGroq,
		ContextWindow:           131072,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "llama-3.1",
	})
	r.Register(ModelInfo{
		ID:                      "llama-3.1-8b-instant",
		Name:                    "Llama 3.1 8B Instant",
		Provider:                ProviderGroq,
		ContextWindow:           131072,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "llama-3.1",
	})
	r.Register(ModelInfo{
		ID:                      "mixtral-8x7b-32768",
		Name:                    "Mixtral 8x7B",
		Provider:                ProviderGroq,
		ContextWindow:           32768,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             false,
		Family:                  "mixtral",
	})

	// X.AI Models
	r.Register(ModelInfo{
		ID:                      "grok-2-latest",
		Name:                    "Grok 2",
		Provider:                ProviderXAI,
		ContextWindow:           131072,
		MaxOutputTokens:         8192,
		SupportsVision:          true,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "grok-2",
	})
	r.Register(ModelInfo{
		ID:                      "grok-beta",
		Name:                    "Grok Beta",
		Provider:                ProviderXAI,
		ContextWindow:           131072,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: true,
		SupportsStreaming:       true,
		Recommended:             false,
		Family:                  "grok",
	})

	// Local Models
	r.Register(ModelInfo{
		ID:                      "meta-llama/Llama-3.1-8B-Instruct",
		Name:                    "Llama 3.1 8B Instruct (Local)",
		Provider:                ProviderLocal,
		ContextWindow:           131072,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: false,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "llama-3.1",
	})
	r.Register(ModelInfo{
		ID:                      "meta-llama/Llama-3.2-3B-Instruct",
		Name:                    "Llama 3.2 3B Instruct (Local)",
		Provider:                ProviderLocal,
		ContextWindow:           131072,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: false,
		SupportsStreaming:       true,
		Recommended:             true,
		Family:                  "llama-3.2",
	})
	r.Register(ModelInfo{
		ID:                      "mistralai/Mistral-7B-Instruct-v0.3",
		Name:                    "Mistral 7B Instruct (Local)",
		Provider:                ProviderLocal,
		ContextWindow:           32768,
		MaxOutputTokens:         8192,
		SupportsVision:          false,
		SupportsFunctionCalling: false,
		SupportsStreaming:       true,
		Recommended:             false,
		Family:                  "mistral",
	})
}

// DefaultRegistry is the global model registry.
var DefaultRegistry = NewModelRegistry()
