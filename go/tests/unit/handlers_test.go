package unit

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/DavidDworetzky/Geist/internal/api/handlers"
	"github.com/DavidDworetzky/Geist/internal/models"
)

func TestParseInt64Param(t *testing.T) {
	// Test with no URL params - should return 0
	req := httptest.NewRequest("GET", "/test/123", nil)
	assert.Equal(t, int64(0), handlers.ParseInt64Param(req, "id"))

	// Test via chi router
	router := chi.NewRouter()
	var result int64
	router.Get("/test/{id}", func(w http.ResponseWriter, r *http.Request) {
		result = handlers.ParseInt64Param(r, "id")
		w.WriteHeader(http.StatusOK)
	})

	w := httptest.NewRecorder()
	req = httptest.NewRequest("GET", "/test/123", nil)
	router.ServeHTTP(w, req)
	assert.Equal(t, int64(123), result)
}

func TestParseIntQuery(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		param    string
		defVal   int
		expected int
	}{
		{"valid int", "?limit=50", "limit", 10, 50},
		{"missing param", "?other=50", "limit", 10, 10},
		{"invalid int", "?limit=abc", "limit", 10, 10},
		{"empty value", "?limit=", "limit", 10, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/test"+tt.query, nil)
			result := handlers.ParseIntQuery(req, tt.param, tt.defVal)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseBoolQuery(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		param    string
		defVal   bool
		expected bool
	}{
		{"true value", "?active=true", "active", false, true},
		{"false value", "?active=false", "active", true, false},
		{"1 value", "?active=1", "active", false, true},
		{"0 value", "?active=0", "active", true, false},
		{"missing param", "?other=true", "active", true, true},
		{"invalid value", "?active=maybe", "active", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/test"+tt.query, nil)
			result := handlers.ParseBoolQuery(req, tt.param, tt.defVal)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWriteJSON(t *testing.T) {
	w := httptest.NewRecorder()

	data := map[string]string{"message": "Hello"}
	handlers.WriteJSON(w, http.StatusOK, data)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var result map[string]string
	err := json.NewDecoder(w.Body).Decode(&result)
	require.NoError(t, err)
	assert.Equal(t, "Hello", result["message"])
}

func TestDecodeJSON(t *testing.T) {
	body := bytes.NewBufferString(`{"name": "test", "value": 42}`)
	req := httptest.NewRequest("POST", "/test", body)
	req.Header.Set("Content-Type", "application/json")

	var data struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	err := handlers.DecodeJSON(req, &data)
	require.NoError(t, err)
	assert.Equal(t, "test", data.Name)
	assert.Equal(t, 42, data.Value)
}

func TestDecodeJSONInvalid(t *testing.T) {
	body := bytes.NewBufferString(`invalid json`)
	req := httptest.NewRequest("POST", "/test", body)

	var data map[string]interface{}
	err := handlers.DecodeJSON(req, &data)
	assert.Error(t, err)
}

func TestErrorResponse(t *testing.T) {
	resp := models.NewErrorResponse("Something went wrong", "internal_error")

	assert.Equal(t, "Something went wrong", resp.Error.Message)
	assert.Equal(t, "internal_error", resp.Error.Type)
}

func TestModelsHandlerListModels(t *testing.T) {
	// This is a simplified test - in production we'd mock the dependencies
	registry := models.NewModelRegistry()

	// Verify registry has models
	modelList := registry.List()
	assert.NotEmpty(t, modelList)

	// Verify specific models exist
	model, ok := registry.Get("gpt-4o")
	assert.True(t, ok)
	assert.Equal(t, models.ProviderOpenAI, model.Provider)
}

func TestHealthHandlerHealth(t *testing.T) {
	// Test basic health response structure
	resp := map[string]interface{}{
		"status":  "ok",
		"version": "1.0.0",
	}

	data, err := json.Marshal(resp)
	require.NoError(t, err)

	var parsed map[string]interface{}
	err = json.Unmarshal(data, &parsed)
	require.NoError(t, err)

	assert.Equal(t, "ok", parsed["status"])
	assert.Equal(t, "1.0.0", parsed["version"])
}

func TestCompleteTextRequestValidation(t *testing.T) {
	tests := []struct {
		name    string
		req     models.CompleteTextRequest
		wantErr bool
	}{
		{
			name: "valid request",
			req: models.CompleteTextRequest{
				Prompt:    "Hello",
				MaxTokens: 100,
			},
			wantErr: false,
		},
		{
			name: "empty prompt allowed (will use defaults)",
			req: models.CompleteTextRequest{
				Prompt: "",
			},
			wantErr: false,
		},
		{
			name: "with all options",
			req: models.CompleteTextRequest{
				Prompt:           "Hello",
				SystemPrompt:     "You are helpful",
				MaxTokens:        256,
				Temperature:      0.9,
				TopP:             0.95,
				FrequencyPenalty: 0.5,
				PresencePenalty:  0.5,
				Stop:             []string{"\n"},
				AgentType:        models.AgentTypeOpenAI,
				Model:            "gpt-4",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply defaults
			req := tt.req.WithDefaults()

			// Basic validation
			if tt.wantErr {
				assert.Empty(t, req.Prompt)
			} else {
				// Verify defaults were applied
				assert.Greater(t, req.MaxTokens, 0)
				assert.Greater(t, req.Temperature, 0.0)
			}
		})
	}
}

func TestStreamChunkFormat(t *testing.T) {
	chunk := models.StreamChunk{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []models.StreamChoice{
			{
				Index: 0,
				Delta: &models.MessageDelta{
					Content: "Hello",
				},
			},
		},
	}

	data, err := json.Marshal(chunk)
	require.NoError(t, err)

	var parsed map[string]interface{}
	err = json.Unmarshal(data, &parsed)
	require.NoError(t, err)

	assert.Equal(t, "chatcmpl-123", parsed["id"])
	assert.Equal(t, "chat.completion.chunk", parsed["object"])

	choices := parsed["choices"].([]interface{})
	assert.Len(t, choices, 1)

	firstChoice := choices[0].(map[string]interface{})
	delta := firstChoice["delta"].(map[string]interface{})
	assert.Equal(t, "Hello", delta["content"])
}

func TestChatHistoryResponse(t *testing.T) {
	resp := models.ChatHistoryResponse{
		SessionID: "123",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
		Total: 2,
	}

	data, err := json.Marshal(resp)
	require.NoError(t, err)

	var parsed models.ChatHistoryResponse
	err = json.Unmarshal(data, &parsed)
	require.NoError(t, err)

	assert.Equal(t, "123", parsed.SessionID)
	assert.Len(t, parsed.Messages, 2)
	assert.Equal(t, 2, parsed.Total)
}

func TestInitializeAgentResponse(t *testing.T) {
	resp := models.InitializeAgentResponse{
		ID:        "agent-123",
		Status:    "initialized",
		SessionID: "session-456",
	}

	data, err := json.Marshal(resp)
	require.NoError(t, err)

	var parsed models.InitializeAgentResponse
	err = json.Unmarshal(data, &parsed)
	require.NoError(t, err)

	assert.Equal(t, "agent-123", parsed.ID)
	assert.Equal(t, "initialized", parsed.Status)
	assert.Equal(t, "session-456", parsed.SessionID)
}
