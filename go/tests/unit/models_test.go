package unit

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/DavidDworetzky/Geist/internal/models"
)

func TestCompleteTextRequestWithDefaults(t *testing.T) {
	req := models.CompleteTextRequest{
		Prompt: "Hello",
	}

	req = req.WithDefaults()

	assert.Equal(t, 256, req.MaxTokens)
	assert.Equal(t, 1, req.N)
	assert.Equal(t, 0.7, req.Temperature)
	assert.Equal(t, 1.0, req.TopP)
	assert.Equal(t, "text", req.ResponseFormat)
	assert.Equal(t, models.AgentTypeLocal, req.AgentType)
}

func TestCompleteTextRequestPreservesValues(t *testing.T) {
	req := models.CompleteTextRequest{
		Prompt:      "Hello",
		MaxTokens:   512,
		Temperature: 0.9,
		TopP:        0.95,
		AgentType:   models.AgentTypeOpenAI,
	}

	req = req.WithDefaults()

	assert.Equal(t, 512, req.MaxTokens)
	assert.Equal(t, 0.9, req.Temperature)
	assert.Equal(t, 0.95, req.TopP)
	assert.Equal(t, models.AgentTypeOpenAI, req.AgentType)
}

func TestNewCompleteTextResponse(t *testing.T) {
	usage := models.Usage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}

	resp := models.NewCompleteTextResponse(
		"test-id",
		"gpt-4",
		"Hello, world!",
		"stop",
		usage,
	)

	assert.Equal(t, "test-id", resp.ID)
	assert.Equal(t, "chat.completion", resp.Object)
	assert.Equal(t, "gpt-4", resp.Model)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, "Hello, world!", resp.Choices[0].Message.Content)
	assert.Equal(t, "assistant", resp.Choices[0].Message.Role)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, 30, resp.Usage.TotalTokens)
	assert.True(t, resp.Created > 0)
}

func TestCompleteTextResponseToJSON(t *testing.T) {
	usage := models.Usage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}

	resp := models.NewCompleteTextResponse(
		"test-id",
		"gpt-4",
		"Hello",
		"stop",
		usage,
	)

	data, err := resp.ToJSON()
	require.NoError(t, err)

	var parsed map[string]interface{}
	err = json.Unmarshal(data, &parsed)
	require.NoError(t, err)

	assert.Equal(t, "test-id", parsed["id"])
	assert.Equal(t, "chat.completion", parsed["object"])
}

func TestModelRegistry(t *testing.T) {
	registry := models.NewModelRegistry()

	// Test default models are registered
	modelList := registry.List()
	assert.NotEmpty(t, modelList)

	// Test Get
	model, ok := registry.Get("gpt-4o")
	assert.True(t, ok)
	assert.Equal(t, "GPT-4o", model.Name)
	assert.Equal(t, models.ProviderOpenAI, model.Provider)

	// Test non-existent model
	_, ok = registry.Get("non-existent")
	assert.False(t, ok)

	// Test ListByProvider
	openaiModels := registry.ListByProvider(models.ProviderOpenAI)
	assert.NotEmpty(t, openaiModels)
	for _, m := range openaiModels {
		assert.Equal(t, models.ProviderOpenAI, m.Provider)
	}

	// Test Register
	customModel := models.ModelInfo{
		ID:       "custom-model",
		Name:     "Custom Model",
		Provider: models.ProviderLocal,
	}
	registry.Register(customModel)

	retrieved, ok := registry.Get("custom-model")
	assert.True(t, ok)
	assert.Equal(t, "Custom Model", retrieved.Name)
}

func TestGetProviderForModel(t *testing.T) {
	tests := []struct {
		modelID  string
		expected models.Provider
		wantErr  bool
	}{
		{"gpt-4o", models.ProviderOpenAI, false},
		{"gpt-4-turbo", models.ProviderOpenAI, false},
		{"o1-preview", models.ProviderOpenAI, false},
		{"claude-3-opus", models.ProviderAnthropic, false},
		{"claude-3-5-sonnet", models.ProviderAnthropic, false},
		{"llama-3.1-70b", models.ProviderGroq, false},
		{"mixtral-8x7b", models.ProviderGroq, false},
		{"grok-2-latest", models.ProviderXAI, false},
		{"meta-llama/Llama-3.1-8B", models.ProviderLocal, false},
		{"unknown-model", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			provider, err := models.GetProviderForModel(tt.modelID)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, provider)
			}
		})
	}
}

func TestGetEndpoint(t *testing.T) {
	tests := []struct {
		provider models.Provider
		expected string
	}{
		{models.ProviderOpenAI, "https://api.openai.com/v1/chat/completions"},
		{models.ProviderAnthropic, "https://api.anthropic.com/v1/messages"},
		{models.ProviderGroq, "https://api.groq.com/openai/v1/chat/completions"},
		{models.ProviderXAI, "https://api.x.ai/v1/chat/completions"},
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			endpoint := models.GetEndpoint(tt.provider)
			assert.Equal(t, tt.expected, endpoint)
		})
	}
}

func TestAgentSettingsDefaults(t *testing.T) {
	settings := models.DefaultAgentSettings()

	assert.Equal(t, "meta-llama/Llama-3.1-8B-Instruct", settings.Model)
	assert.Equal(t, "You are a helpful AI assistant.", settings.SystemPrompt)
	assert.Equal(t, 2048, settings.MaxTokens)
	assert.Equal(t, 0.7, settings.Temperature)
	assert.Equal(t, 1.0, settings.TopP)
}

func TestAgentSettingsMerge(t *testing.T) {
	base := models.DefaultAgentSettings()

	override := models.AgentSettings{
		Model:       "custom-model",
		Temperature: 0.9,
	}

	merged := base.Merge(override)

	assert.Equal(t, "custom-model", merged.Model)
	assert.Equal(t, 0.9, merged.Temperature)
	// These should be unchanged
	assert.Equal(t, base.SystemPrompt, merged.SystemPrompt)
	assert.Equal(t, base.MaxTokens, merged.MaxTokens)
	assert.Equal(t, base.TopP, merged.TopP)
}

func TestAgentContext(t *testing.T) {
	settings := models.DefaultAgentSettings()
	ctx := models.NewAgentContext("test-id", settings)

	assert.Equal(t, "test-id", ctx.ID)
	assert.Equal(t, settings, ctx.Settings)
	assert.Empty(t, ctx.WorldContext)
	assert.Empty(t, ctx.TaskContext)
	assert.Empty(t, ctx.ExecutionContext)
	assert.NotZero(t, ctx.CreatedAt)

	// Test adding context
	ctx.AddWorldContext("world info")
	ctx.AddTaskContext("task info")
	ctx.AddExecutionContext("execution info")

	assert.Len(t, ctx.WorldContext, 1)
	assert.Len(t, ctx.TaskContext, 1)
	assert.Len(t, ctx.ExecutionContext, 1)
}

func TestChatMessage(t *testing.T) {
	msg := models.ChatMessage{
		ID:        1,
		SessionID: 100,
		Role:      "user",
		Content:   "Hello",
		CreatedAt: time.Now(),
	}

	// Test ToMessage
	converted := msg.ToMessage()
	assert.Equal(t, "user", converted.Role)
	assert.Equal(t, "Hello", converted.Content)

	// Test MetadataJSON with empty metadata
	meta, err := msg.MetadataJSON()
	assert.NoError(t, err)
	assert.Nil(t, meta)

	// Test MetadataJSON with valid metadata
	msg.Metadata = `{"key": "value"}`
	meta, err = msg.MetadataJSON()
	assert.NoError(t, err)
	assert.Equal(t, "value", meta["key"])
}

func TestUserSettingsDefaults(t *testing.T) {
	settings := models.DefaultUserSettings(123)

	assert.Equal(t, int64(123), settings.UserID)
	assert.Equal(t, models.AgentTypeLocal, settings.DefaultAgentType)
	assert.Equal(t, "meta-llama/Llama-3.1-8B-Instruct", settings.DefaultModel)
	assert.True(t, settings.Preferences.StreamResponses)
	assert.True(t, settings.Preferences.SaveChatHistory)
}

func TestStreamChunkToSSE(t *testing.T) {
	chunk := models.StreamChunk{
		ID:      "test-id",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []models.StreamChoice{
			{
				Index: 0,
				Delta: &models.MessageDelta{Content: "Hello"},
			},
		},
	}

	data, err := models.StreamChunkToSSE(chunk)
	require.NoError(t, err)

	assert.True(t, len(data) > 0)
	assert.Contains(t, string(data), "data: ")
	assert.Contains(t, string(data), "test-id")
	assert.Contains(t, string(data), "Hello")
}
