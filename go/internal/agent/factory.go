// Package agent provides the core agent interface and implementations.
package agent

import (
	"fmt"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/DavidDworetzky/Geist/internal/config"
	"github.com/DavidDworetzky/Geist/internal/models"
)

// Factory creates and manages agent instances.
type Factory struct {
	cfg    *config.Config
	cache  map[string]Agent
	mu     sync.RWMutex
	pool   *InferencePool
}

// NewFactory creates a new agent factory.
func NewFactory(cfg *config.Config) *Factory {
	return &Factory{
		cfg:   cfg,
		cache: make(map[string]Agent),
	}
}

// Initialize initializes the factory, including connecting to the inference service.
func (f *Factory) Initialize() error {
	pool, err := NewInferencePool(f.cfg.MLX.Address, f.cfg.Models.DefaultModel, f.cfg.MLX.ConnectionPool)
	if err != nil {
		log.Warn().Err(err).Msg("Failed to initialize inference pool, local agents will not be available")
	} else {
		f.pool = pool
	}
	return nil
}

// Close closes all cached agents and the inference pool.
func (f *Factory) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	for key, agent := range f.cache {
		if err := agent.Close(); err != nil {
			log.Error().Err(err).Str("key", key).Msg("Error closing agent")
		}
	}
	f.cache = make(map[string]Agent)

	if f.pool != nil {
		return f.pool.Close()
	}
	return nil
}

// CreateAgent creates a new agent based on the request parameters.
func (f *Factory) CreateAgent(agentType models.AgentType, model string) (Agent, error) {
	// Determine provider from model if not explicitly specified
	if agentType == "" || agentType == models.AgentTypeOnline {
		provider, err := models.GetProviderForModel(model)
		if err != nil {
			return nil, fmt.Errorf("determining provider: %w", err)
		}
		agentType = models.AgentType(provider)
	}

	switch agentType {
	case models.AgentTypeLocal, models.AgentTypeLlama:
		return f.createLocalAgent(model)
	case models.AgentTypeOpenAI:
		return f.createOnlineAgent(models.ProviderOpenAI, model)
	case models.AgentTypeAnthropic:
		return f.createOnlineAgent(models.ProviderAnthropic, model)
	case models.AgentTypeGroq:
		return f.createOnlineAgent(models.ProviderGroq, model)
	case models.AgentTypeXAI:
		return f.createOnlineAgent(models.ProviderXAI, model)
	default:
		return nil, fmt.Errorf("unsupported agent type: %s", agentType)
	}
}

// GetAgent gets or creates an agent, using caching for efficiency.
func (f *Factory) GetAgent(agentType models.AgentType, model string) (Agent, error) {
	cacheKey := fmt.Sprintf("%s:%s", agentType, model)

	// Try to get from cache first
	f.mu.RLock()
	if agent, ok := f.cache[cacheKey]; ok {
		f.mu.RUnlock()
		return agent, nil
	}
	f.mu.RUnlock()

	// Create new agent
	agent, err := f.CreateAgent(agentType, model)
	if err != nil {
		return nil, err
	}

	// Cache it
	f.mu.Lock()
	f.cache[cacheKey] = agent
	f.mu.Unlock()

	return agent, nil
}

// createLocalAgent creates a local agent using the inference pool.
func (f *Factory) createLocalAgent(model string) (Agent, error) {
	if f.pool == nil {
		return nil, fmt.Errorf("inference pool not initialized")
	}

	// For local agents, we return from the pool
	// The model parameter is used to override the default model
	agent := f.pool.Get()
	if model != "" && model != agent.Model() {
		// Create a new agent with the specified model
		return NewLocalAgent(f.cfg.MLX.Address, model)
	}
	return agent, nil
}

// createOnlineAgent creates an online agent for the specified provider.
func (f *Factory) createOnlineAgent(provider models.Provider, model string) (Agent, error) {
	var apiKey string

	switch provider {
	case models.ProviderOpenAI:
		apiKey = f.cfg.APIKeys.OpenAI
		if apiKey == "" {
			return nil, fmt.Errorf("OpenAI API key not configured")
		}
	case models.ProviderAnthropic:
		apiKey = f.cfg.APIKeys.Anthropic
		if apiKey == "" {
			return nil, fmt.Errorf("Anthropic API key not configured")
		}
	case models.ProviderGroq:
		apiKey = f.cfg.APIKeys.Groq
		if apiKey == "" {
			return nil, fmt.Errorf("Groq API key not configured")
		}
	case models.ProviderXAI:
		apiKey = f.cfg.APIKeys.XAI
		if apiKey == "" {
			return nil, fmt.Errorf("X.AI API key not configured")
		}
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}

	return NewOnlineAgent(provider, model, apiKey), nil
}

// GetDefaultAgent returns the default agent based on configuration.
func (f *Factory) GetDefaultAgent() (Agent, error) {
	return f.GetAgent(models.AgentTypeLocal, f.cfg.Models.DefaultModel)
}

// ListAvailableProviders returns a list of providers with configured API keys.
func (f *Factory) ListAvailableProviders() []models.Provider {
	var providers []models.Provider

	// Local is always available if inference pool is initialized
	if f.pool != nil {
		providers = append(providers, models.ProviderLocal)
	}

	if f.cfg.HasOpenAI() {
		providers = append(providers, models.ProviderOpenAI)
	}
	if f.cfg.HasAnthropic() {
		providers = append(providers, models.ProviderAnthropic)
	}
	if f.cfg.HasGroq() {
		providers = append(providers, models.ProviderGroq)
	}
	if f.cfg.HasXAI() {
		providers = append(providers, models.ProviderXAI)
	}

	return providers
}

// AgentFromRequest creates an agent from a completion request.
func (f *Factory) AgentFromRequest(req *models.CompleteTextRequest) (Agent, error) {
	agentType := req.AgentType
	model := req.Model

	// Use defaults if not specified
	if agentType == "" {
		agentType = models.AgentTypeLocal
	}
	if model == "" {
		model = f.cfg.Models.DefaultModel
	}

	return f.GetAgent(agentType, model)
}
