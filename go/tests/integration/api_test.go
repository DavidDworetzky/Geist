// Package integration provides integration tests for the Geist server.
package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/DavidDworetzky/Geist/internal/agent"
	"github.com/DavidDworetzky/Geist/internal/api"
	"github.com/DavidDworetzky/Geist/internal/config"
	"github.com/DavidDworetzky/Geist/internal/database"
	"github.com/DavidDworetzky/Geist/internal/models"
)

// testServer creates a test server for integration tests.
type testServer struct {
	server  *api.Server
	handler http.Handler
}

func newTestServer(t *testing.T) *testServer {
	t.Helper()

	// Set required env vars for config
	os.Setenv("POSTGRES_PASSWORD", "testpass")
	os.Setenv("DB_HOST", "localhost")

	cfg, err := config.Load()
	require.NoError(t, err)

	// For integration tests without a real DB, we'll use nil
	// In a real setup, you'd use testcontainers or a test database
	factory := agent.NewFactory(cfg)

	// Create server without database for these tests
	// In production, you'd initialize the database properly
	server := api.NewServer(cfg, nil, factory)

	return &testServer{
		server:  server,
		handler: server.Handler(),
	}
}

func (ts *testServer) doRequest(t *testing.T, method, path string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()

	var reqBody *bytes.Buffer
	if body != nil {
		data, err := json.Marshal(body)
		require.NoError(t, err)
		reqBody = bytes.NewBuffer(data)
	} else {
		reqBody = bytes.NewBuffer(nil)
	}

	req := httptest.NewRequest(method, path, reqBody)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	ts.handler.ServeHTTP(w, req)

	return w
}

func TestHealthEndpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	w := ts.doRequest(t, "GET", "/health", nil)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.NewDecoder(w.Body).Decode(&resp)
	require.NoError(t, err)

	assert.Equal(t, "ok", resp["status"])
}

func TestLiveEndpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	w := ts.doRequest(t, "GET", "/live", nil)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	err := json.NewDecoder(w.Body).Decode(&resp)
	require.NoError(t, err)

	assert.Equal(t, "alive", resp["status"])
}

func TestModelsEndpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	w := ts.doRequest(t, "GET", "/api/v1/models", nil)

	// Should return 200 even without database
	assert.Equal(t, http.StatusOK, w.Code)

	var resp struct {
		Object string             `json:"object"`
		Data   []models.ModelInfo `json:"data"`
	}
	err := json.NewDecoder(w.Body).Decode(&resp)
	require.NoError(t, err)

	assert.Equal(t, "list", resp.Object)
}

func TestProvidersEndpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	w := ts.doRequest(t, "GET", "/api/v1/providers", nil)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp struct {
		Providers []string `json:"providers"`
	}
	err := json.NewDecoder(w.Body).Decode(&resp)
	require.NoError(t, err)

	// Should have at least some providers
	assert.NotNil(t, resp.Providers)
}

func TestCORSHeaders(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	req := httptest.NewRequest("OPTIONS", "/health", nil)
	req.Header.Set("Origin", "http://localhost:3000")
	req.Header.Set("Access-Control-Request-Method", "GET")

	w := httptest.NewRecorder()
	ts.handler.ServeHTTP(w, req)

	// Should handle preflight
	assert.Contains(t, []int{http.StatusOK, http.StatusNoContent}, w.Code)
}

func TestRequestIDHeader(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	w := ts.doRequest(t, "GET", "/health", nil)

	// Should have request ID header
	requestID := w.Header().Get("X-Request-ID")
	assert.NotEmpty(t, requestID)
}

func TestCompleteTextEndpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	req := models.CompleteTextRequest{
		Prompt:      "Hello",
		MaxTokens:   10,
		Temperature: 0.7,
		AgentType:   models.AgentTypeLocal,
	}

	w := ts.doRequest(t, "POST", "/agent/complete_text", req)

	// Without a real inference backend, this might fail
	// but we can verify the endpoint exists and processes the request
	assert.Contains(t, []int{http.StatusOK, http.StatusInternalServerError}, w.Code)
}

func TestInvalidJSON(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	req := httptest.NewRequest("POST", "/agent/complete_text", bytes.NewBufferString("invalid json"))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	ts.handler.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestContextCancellation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ts := newTestServer(t)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	req := httptest.NewRequest("GET", "/health", nil).WithContext(ctx)
	w := httptest.NewRecorder()
	ts.handler.ServeHTTP(w, req)

	// Request should complete (health is fast) or be cancelled
	assert.Contains(t, []int{http.StatusOK, http.StatusServiceUnavailable}, w.Code)
}

// TestDatabaseIntegration tests database operations
// This requires a real database connection
func TestDatabaseIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping database integration test in short mode")
	}

	// Skip if no database URL is set
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL not set, skipping database integration test")
	}

	os.Setenv("POSTGRES_PASSWORD", "testpass")
	cfg, err := config.Load()
	require.NoError(t, err)

	ctx := context.Background()
	db, err := database.New(ctx, cfg.Database)
	if err != nil {
		t.Skipf("Could not connect to database: %v", err)
	}
	defer db.Close()

	// Test ping
	err = db.Ping(ctx)
	assert.NoError(t, err)

	// Test health
	health, err := db.Health(ctx)
	require.NoError(t, err)
	assert.Equal(t, "up", health["status"])
}

// TestAgentFactory tests the agent factory
func TestAgentFactory(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	os.Setenv("POSTGRES_PASSWORD", "testpass")
	os.Setenv("OPENAI_API_KEY", "test-key")
	defer os.Unsetenv("OPENAI_API_KEY")

	cfg, err := config.Load()
	require.NoError(t, err)

	factory := agent.NewFactory(cfg)
	defer factory.Close()

	// Test listing providers
	providers := factory.ListAvailableProviders()
	assert.Contains(t, providers, models.ProviderOpenAI)

	// Test creating online agent
	ag, err := factory.CreateAgent(models.AgentTypeOpenAI, "gpt-4")
	require.NoError(t, err)
	assert.NotNil(t, ag)
	assert.Equal(t, "gpt-4", ag.Model())
}

// BenchmarkHealthEndpoint benchmarks the health endpoint
func BenchmarkHealthEndpoint(b *testing.B) {
	os.Setenv("POSTGRES_PASSWORD", "testpass")
	cfg, _ := config.Load()
	factory := agent.NewFactory(cfg)
	server := api.NewServer(cfg, nil, factory)
	handler := server.Handler()

	req := httptest.NewRequest("GET", "/health", nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
	}
}

// BenchmarkModelsEndpoint benchmarks the models endpoint
func BenchmarkModelsEndpoint(b *testing.B) {
	os.Setenv("POSTGRES_PASSWORD", "testpass")
	cfg, _ := config.Load()
	factory := agent.NewFactory(cfg)
	server := api.NewServer(cfg, nil, factory)
	handler := server.Handler()

	req := httptest.NewRequest("GET", "/api/v1/models", nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
	}
}
