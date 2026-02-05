package unit

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/DavidDworetzky/Geist/internal/agent"
	"github.com/DavidDworetzky/Geist/internal/models"
)

func TestCompletionRequestWithDefaults(t *testing.T) {
	req := agent.CompletionRequest{
		Prompt: "Hello",
	}

	req = req.WithDefaults()

	assert.Equal(t, 256, req.MaxTokens)
	assert.Equal(t, 0.7, req.Temperature)
	assert.Equal(t, 1.0, req.TopP)
}

func TestOnlineAgentOpenAI(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer test-key")

		// Parse body
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		assert.Equal(t, "gpt-4", body["model"])
		assert.NotEmpty(t, body["messages"])

		// Return mock response
		resp := map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "gpt-4",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": "Hello! How can I help you?",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     10,
				"completion_tokens": 8,
				"total_tokens":      18,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create agent with mock endpoint using the test constructor
	ag := agent.NewOnlineAgentWithEndpoint(models.ProviderOpenAI, "gpt-4", "test-key", server.URL)

	ctx := context.Background()
	req := &agent.CompletionRequest{
		Prompt:       "Hello",
		SystemPrompt: "You are helpful",
		MaxTokens:    100,
		Temperature:  0.7,
	}

	resp, err := ag.CompleteText(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, "chatcmpl-123", resp.ID)
	assert.Equal(t, "Hello! How can I help you?", resp.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, 18, resp.Usage.TotalTokens)
}

func TestOnlineAgentType(t *testing.T) {
	ag := agent.NewOnlineAgent(models.ProviderOpenAI, "gpt-4", "test-key")
	assert.Equal(t, models.AgentType("openai"), ag.Type())

	ag2 := agent.NewOnlineAgent(models.ProviderAnthropic, "claude-3", "test-key")
	assert.Equal(t, models.AgentType("anthropic"), ag2.Type())
}

func TestOnlineAgentModel(t *testing.T) {
	ag := agent.NewOnlineAgent(models.ProviderOpenAI, "gpt-4-turbo", "test-key")
	assert.Equal(t, "gpt-4-turbo", ag.Model())
}

func TestOnlineAgentClose(t *testing.T) {
	ag := agent.NewOnlineAgent(models.ProviderOpenAI, "gpt-4", "test-key")
	err := ag.Close()
	assert.NoError(t, err)
}

func TestOnlineAgentInitialize(t *testing.T) {
	ag := agent.NewOnlineAgent(models.ProviderOpenAI, "gpt-4", "test-key")
	err := ag.Initialize(context.Background(), "Test task")
	assert.NoError(t, err)
}

func TestOnlineAgentTick(t *testing.T) {
	ag := agent.NewOnlineAgent(models.ProviderOpenAI, "gpt-4", "test-key")
	result, err := ag.Tick(context.Background())
	require.NoError(t, err)
	assert.Equal(t, "idle", result.Status)
	assert.True(t, result.Complete)
}

func TestStreamWriter(t *testing.T) {
	var buf bytes.Buffer
	sw := agent.NewStreamWriter(&buf)

	n, err := sw.Write([]byte("Hello"))
	assert.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, "Hello", buf.String())
}

func TestStreamWriterWithFlusher(t *testing.T) {
	// Create a mock flusher
	flusher := &mockFlusher{Buffer: new(bytes.Buffer)}
	sw := agent.NewStreamWriter(flusher)

	_, err := sw.Write([]byte("Hello"))
	assert.NoError(t, err)
	assert.True(t, flusher.flushed)
}

type mockFlusher struct {
	*bytes.Buffer
	flushed bool
}

func (m *mockFlusher) Flush() {
	m.flushed = true
}

func TestStreamWriterWriteToken(t *testing.T) {
	var buf bytes.Buffer
	sw := agent.NewStreamWriter(&buf)

	token := agent.StreamToken{
		Text:    "Hello",
		IsFinal: false,
	}

	err := sw.WriteToken(token)
	assert.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "data: ")
	assert.Contains(t, output, "Hello")
}

func TestGenerateID(t *testing.T) {
	id1 := agent.GenerateID()
	id2 := agent.GenerateID()

	assert.NotEmpty(t, id1)
	assert.NotEmpty(t, id2)
	assert.NotEqual(t, id1, id2)
	assert.Contains(t, id1, "chatcmpl-")
}

func TestStreamToken(t *testing.T) {
	// Test non-final token
	token := agent.StreamToken{
		Text:    "Hello",
		IsFinal: false,
	}
	assert.Equal(t, "Hello", token.Text)
	assert.False(t, token.IsFinal)
	assert.Empty(t, token.FinishReason)

	// Test final token
	finalToken := agent.StreamToken{
		Text:         "",
		IsFinal:      true,
		FinishReason: "stop",
	}
	assert.True(t, finalToken.IsFinal)
	assert.Equal(t, "stop", finalToken.FinishReason)

	// Test error token
	errToken := agent.StreamToken{
		Error:   io.EOF,
		IsFinal: true,
	}
	assert.Error(t, errToken.Error)
	assert.True(t, errToken.IsFinal)
}

func TestTickResult(t *testing.T) {
	result := agent.TickResult{
		Status:   "running",
		Messages: []string{"Processing step 1", "Processing step 2"},
		Complete: false,
	}

	assert.Equal(t, "running", result.Status)
	assert.Len(t, result.Messages, 2)
	assert.False(t, result.Complete)

	// Test completed result
	completedResult := agent.TickResult{
		Status:   "completed",
		Complete: true,
	}
	assert.True(t, completedResult.Complete)
}

func TestUsage(t *testing.T) {
	usage := agent.Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}

	assert.Equal(t, 100, usage.PromptTokens)
	assert.Equal(t, 50, usage.CompletionTokens)
	assert.Equal(t, 150, usage.TotalTokens)
}

func TestCompletionResponse(t *testing.T) {
	resp := agent.CompletionResponse{
		ID:           "test-123",
		Content:      "Hello, world!",
		FinishReason: "stop",
		Model:        "gpt-4",
		Usage: agent.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
		ChatID: 42,
	}

	assert.Equal(t, "test-123", resp.ID)
	assert.Equal(t, "Hello, world!", resp.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, "gpt-4", resp.Model)
	assert.Equal(t, int64(42), resp.ChatID)
	assert.Equal(t, 15, resp.Usage.TotalTokens)
}
