// Package config provides configuration loading and management for the Geist server.
package config

import (
	"fmt"
	"time"

	"github.com/kelseyhightower/envconfig"
)

// Config holds all configuration values for the Geist server.
type Config struct {
	// Server configuration
	Server ServerConfig

	// Database configuration
	Database DatabaseConfig

	// MLX inference service configuration
	MLX MLXConfig

	// API keys for external services
	APIKeys APIKeysConfig

	// Model configuration
	Models ModelsConfig

	// Logging configuration
	Logging LoggingConfig
}

// ServerConfig holds HTTP server configuration.
type ServerConfig struct {
	Host            string        `envconfig:"SERVER_HOST" default:"0.0.0.0"`
	Port            int           `envconfig:"SERVER_PORT" default:"8000"`
	ReadTimeout     time.Duration `envconfig:"SERVER_READ_TIMEOUT" default:"30s"`
	WriteTimeout    time.Duration `envconfig:"SERVER_WRITE_TIMEOUT" default:"120s"`
	IdleTimeout     time.Duration `envconfig:"SERVER_IDLE_TIMEOUT" default:"60s"`
	ShutdownTimeout time.Duration `envconfig:"SERVER_SHUTDOWN_TIMEOUT" default:"30s"`
	MaxRequestSize  int64         `envconfig:"SERVER_MAX_REQUEST_SIZE" default:"10485760"` // 10MB
}

// DatabaseConfig holds database connection configuration.
type DatabaseConfig struct {
	Host            string        `envconfig:"DB_HOST" default:"localhost"`
	Port            int           `envconfig:"DB_PORT" default:"5432"`
	Name            string        `envconfig:"POSTGRES_DB" default:"geist"`
	User            string        `envconfig:"POSTGRES_USER" default:"geist"`
	Password        string        `envconfig:"POSTGRES_PASSWORD" required:"true"`
	SSLMode         string        `envconfig:"DB_SSL_MODE" default:"disable"`
	MaxOpenConns    int           `envconfig:"DB_MAX_OPEN_CONNS" default:"25"`
	MaxIdleConns    int           `envconfig:"DB_MAX_IDLE_CONNS" default:"5"`
	ConnMaxLifetime time.Duration `envconfig:"DB_CONN_MAX_LIFETIME" default:"5m"`
	ConnMaxIdleTime time.Duration `envconfig:"DB_CONN_MAX_IDLE_TIME" default:"1m"`
}

// DSN returns the PostgreSQL connection string.
func (c DatabaseConfig) DSN() string {
	return fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		c.Host, c.Port, c.User, c.Password, c.Name, c.SSLMode,
	)
}

// URL returns the PostgreSQL connection URL.
func (c DatabaseConfig) URL() string {
	return fmt.Sprintf(
		"postgres://%s:%s@%s:%d/%s?sslmode=%s",
		c.User, c.Password, c.Host, c.Port, c.Name, c.SSLMode,
	)
}

// MLXConfig holds MLX inference service configuration.
type MLXConfig struct {
	Address          string        `envconfig:"MLX_SERVICE_ADDR" default:"localhost:50051"`
	Timeout          time.Duration `envconfig:"MLX_TIMEOUT" default:"120s"`
	MaxRetries       int           `envconfig:"MLX_MAX_RETRIES" default:"3"`
	RetryBackoff     time.Duration `envconfig:"MLX_RETRY_BACKOFF" default:"1s"`
	HealthCheckFreq  time.Duration `envconfig:"MLX_HEALTH_CHECK_FREQ" default:"30s"`
	ConnectionPool   int           `envconfig:"MLX_CONNECTION_POOL" default:"10"`
	KeepAliveTime    time.Duration `envconfig:"MLX_KEEPALIVE_TIME" default:"30s"`
	KeepAliveTimeout time.Duration `envconfig:"MLX_KEEPALIVE_TIMEOUT" default:"10s"`
}

// APIKeysConfig holds API keys for external LLM providers.
type APIKeysConfig struct {
	OpenAI      string `envconfig:"OPENAI_API_KEY"`
	Anthropic   string `envconfig:"ANTHROPIC_API_KEY"`
	Groq        string `envconfig:"GROQ_API_KEY"`
	XAI         string `envconfig:"GROK_API_KEY"`
	HuggingFace string `envconfig:"HUGGING_FACE_HUB_TOKEN"`
}

// ModelsConfig holds model-related configuration.
type ModelsConfig struct {
	LocalWeightsDir string `envconfig:"LOCAL_WEIGHTS_DIR" default:"./weights"`
	DefaultModel    string `envconfig:"DEFAULT_MODEL" default:"meta-llama/Llama-3.1-8B-Instruct"`
	DefaultRunner   string `envconfig:"DEFAULT_RUNNER" default:"mlx_llama"`
	CacheDir        string `envconfig:"MODEL_CACHE_DIR" default:"~/.cache/huggingface"`
}

// LoggingConfig holds logging configuration.
type LoggingConfig struct {
	Level           string `envconfig:"LOG_LEVEL" default:"info"`
	Format          string `envconfig:"LOG_FORMAT" default:"json"`
	Enhanced        bool   `envconfig:"ENHANCED_LOGGING" default:"false"`
	IncludeCaller   bool   `envconfig:"LOG_INCLUDE_CALLER" default:"false"`
	OutputPath      string `envconfig:"LOG_OUTPUT_PATH" default:"stdout"`
	RequestLogging  bool   `envconfig:"LOG_REQUESTS" default:"true"`
	ResponseLogging bool   `envconfig:"LOG_RESPONSES" default:"false"`
}

// Load loads configuration from environment variables.
func Load() (*Config, error) {
	var cfg Config

	if err := envconfig.Process("", &cfg.Server); err != nil {
		return nil, fmt.Errorf("loading server config: %w", err)
	}

	if err := envconfig.Process("", &cfg.Database); err != nil {
		return nil, fmt.Errorf("loading database config: %w", err)
	}

	if err := envconfig.Process("", &cfg.MLX); err != nil {
		return nil, fmt.Errorf("loading mlx config: %w", err)
	}

	if err := envconfig.Process("", &cfg.APIKeys); err != nil {
		return nil, fmt.Errorf("loading api keys config: %w", err)
	}

	if err := envconfig.Process("", &cfg.Models); err != nil {
		return nil, fmt.Errorf("loading models config: %w", err)
	}

	if err := envconfig.Process("", &cfg.Logging); err != nil {
		return nil, fmt.Errorf("loading logging config: %w", err)
	}

	return &cfg, nil
}

// MustLoad loads configuration and panics on error.
func MustLoad() *Config {
	cfg, err := Load()
	if err != nil {
		panic(fmt.Sprintf("failed to load config: %v", err))
	}
	return cfg
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", c.Server.Port)
	}

	if c.Database.Port < 1 || c.Database.Port > 65535 {
		return fmt.Errorf("invalid database port: %d", c.Database.Port)
	}

	if c.MLX.Timeout < time.Second {
		return fmt.Errorf("mlx timeout too short: %v", c.MLX.Timeout)
	}

	return nil
}

// HasOpenAI returns true if OpenAI API key is configured.
func (c *Config) HasOpenAI() bool {
	return c.APIKeys.OpenAI != ""
}

// HasAnthropic returns true if Anthropic API key is configured.
func (c *Config) HasAnthropic() bool {
	return c.APIKeys.Anthropic != ""
}

// HasGroq returns true if Groq API key is configured.
func (c *Config) HasGroq() bool {
	return c.APIKeys.Groq != ""
}

// HasXAI returns true if X.AI API key is configured.
func (c *Config) HasXAI() bool {
	return c.APIKeys.XAI != ""
}
