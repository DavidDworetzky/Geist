// Package agent provides the core agent interface and implementations.
package agent

import (
	"context"
	"io"

	"github.com/DavidDworetzky/Geist/internal/models"
)

// Agent defines the interface for all agent implementations.
type Agent interface {
	// CompleteText generates a text completion for the given request.
	CompleteText(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// StreamCompleteText generates a streaming text completion.
	// The returned channel will receive tokens as they are generated.
	// The channel will be closed when generation is complete.
	StreamCompleteText(ctx context.Context, req *CompletionRequest) (<-chan StreamToken, error)

	// Initialize prepares the agent with a task prompt.
	Initialize(ctx context.Context, taskPrompt string) error

	// Tick advances the agent's execution loop by one step.
	Tick(ctx context.Context) (*TickResult, error)

	// Type returns the agent type.
	Type() models.AgentType

	// Model returns the model being used.
	Model() string

	// Close releases any resources held by the agent.
	Close() error
}

// CompletionRequest contains all parameters for a completion request.
type CompletionRequest struct {
	Prompt           string
	SystemPrompt     string
	MaxTokens        int
	Temperature      float64
	TopP             float64
	FrequencyPenalty float64
	PresencePenalty  float64
	Stop             []string
	ChatHistory      []models.Message
	ChatID           *int64
}

// WithDefaults returns a copy with default values applied.
func (r CompletionRequest) WithDefaults() CompletionRequest {
	if r.MaxTokens == 0 {
		r.MaxTokens = 256
	}
	if r.Temperature == 0 {
		r.Temperature = 0.7
	}
	if r.TopP == 0 {
		r.TopP = 1.0
	}
	return r
}

// CompletionResponse contains the result of a completion request.
type CompletionResponse struct {
	ID           string
	Content      string
	FinishReason string
	Usage        Usage
	Model        string
	ChatID       int64
}

// Usage contains token usage statistics.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// StreamToken represents a single token in a streaming response.
type StreamToken struct {
	Text         string
	IsFinal      bool
	FinishReason string
	Error        error
}

// TickResult contains the result of an agent tick.
type TickResult struct {
	Status   string
	Messages []string
	Complete bool
}

// StreamWriter wraps an io.Writer for streaming responses.
type StreamWriter struct {
	w       io.Writer
	flusher interface{ Flush() }
}

// NewStreamWriter creates a new StreamWriter.
func NewStreamWriter(w io.Writer) *StreamWriter {
	sw := &StreamWriter{w: w}
	if f, ok := w.(interface{ Flush() }); ok {
		sw.flusher = f
	}
	return sw
}

// Write writes data and flushes if possible.
func (sw *StreamWriter) Write(p []byte) (n int, err error) {
	n, err = sw.w.Write(p)
	if sw.flusher != nil && err == nil {
		sw.flusher.Flush()
	}
	return
}

// WriteToken writes a single token in SSE format.
func (sw *StreamWriter) WriteToken(token StreamToken) error {
	chunk := models.StreamChunk{
		ID:      "chatcmpl-stream",
		Object:  "chat.completion.chunk",
		Model:   "",
		Choices: []models.StreamChoice{{
			Index: 0,
			Delta: &models.MessageDelta{Content: token.Text},
		}},
	}
	if token.IsFinal {
		chunk.Choices[0].FinishReason = token.FinishReason
	}

	data, err := models.StreamChunkToSSE(chunk)
	if err != nil {
		return err
	}
	_, err = sw.Write(data)
	return err
}
