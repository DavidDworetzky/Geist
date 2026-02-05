package unit

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/DavidDworetzky/Geist/internal/config"
)

func TestConfigLoad(t *testing.T) {
	// Set required environment variables
	os.Setenv("POSTGRES_PASSWORD", "testpass")
	defer os.Unsetenv("POSTGRES_PASSWORD")

	cfg, err := config.Load()
	require.NoError(t, err)
	require.NotNil(t, cfg)

	// Check defaults
	assert.Equal(t, "0.0.0.0", cfg.Server.Host)
	assert.Equal(t, 8000, cfg.Server.Port)
	assert.Equal(t, 30*time.Second, cfg.Server.ReadTimeout)

	assert.Equal(t, "localhost", cfg.Database.Host)
	assert.Equal(t, 5432, cfg.Database.Port)
	assert.Equal(t, "geist", cfg.Database.Name)

	assert.Equal(t, "localhost:50051", cfg.MLX.Address)
	assert.Equal(t, 120*time.Second, cfg.MLX.Timeout)

	assert.Equal(t, "meta-llama/Llama-3.1-8B-Instruct", cfg.Models.DefaultModel)
	assert.Equal(t, "mlx_llama", cfg.Models.DefaultRunner)
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(*config.Config)
		wantErr bool
	}{
		{
			name:    "valid config",
			modify:  func(c *config.Config) {},
			wantErr: false,
		},
		{
			name: "invalid server port - too low",
			modify: func(c *config.Config) {
				c.Server.Port = 0
			},
			wantErr: true,
		},
		{
			name: "invalid server port - too high",
			modify: func(c *config.Config) {
				c.Server.Port = 70000
			},
			wantErr: true,
		},
		{
			name: "invalid database port",
			modify: func(c *config.Config) {
				c.Database.Port = -1
			},
			wantErr: true,
		},
		{
			name: "mlx timeout too short",
			modify: func(c *config.Config) {
				c.MLX.Timeout = 100 * time.Millisecond
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			os.Setenv("POSTGRES_PASSWORD", "testpass")
			defer os.Unsetenv("POSTGRES_PASSWORD")

			cfg, err := config.Load()
			require.NoError(t, err)

			tt.modify(cfg)

			err = cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDatabaseConfigDSN(t *testing.T) {
	cfg := config.DatabaseConfig{
		Host:     "localhost",
		Port:     5432,
		User:     "geist",
		Password: "secret",
		Name:     "geist_db",
		SSLMode:  "disable",
	}

	dsn := cfg.DSN()
	assert.Contains(t, dsn, "host=localhost")
	assert.Contains(t, dsn, "port=5432")
	assert.Contains(t, dsn, "user=geist")
	assert.Contains(t, dsn, "password=secret")
	assert.Contains(t, dsn, "dbname=geist_db")
	assert.Contains(t, dsn, "sslmode=disable")
}

func TestDatabaseConfigURL(t *testing.T) {
	cfg := config.DatabaseConfig{
		Host:     "localhost",
		Port:     5432,
		User:     "geist",
		Password: "secret",
		Name:     "geist_db",
		SSLMode:  "disable",
	}

	url := cfg.URL()
	assert.Equal(t, "postgres://geist:secret@localhost:5432/geist_db?sslmode=disable", url)
}

func TestAPIKeyHelpers(t *testing.T) {
	os.Setenv("POSTGRES_PASSWORD", "testpass")
	defer os.Unsetenv("POSTGRES_PASSWORD")

	cfg, err := config.Load()
	require.NoError(t, err)

	// Initially no keys set
	assert.False(t, cfg.HasOpenAI())
	assert.False(t, cfg.HasAnthropic())
	assert.False(t, cfg.HasGroq())
	assert.False(t, cfg.HasXAI())

	// Set keys
	cfg.APIKeys.OpenAI = "sk-test"
	cfg.APIKeys.Anthropic = "sk-ant-test"
	cfg.APIKeys.Groq = "gsk_test"
	cfg.APIKeys.XAI = "xai_test"

	assert.True(t, cfg.HasOpenAI())
	assert.True(t, cfg.HasAnthropic())
	assert.True(t, cfg.HasGroq())
	assert.True(t, cfg.HasXAI())
}
