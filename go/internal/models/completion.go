// Package models provides data models for API requests and responses.
package models

import (
	"encoding/json"
	"time"
)

// AgentType represents the type of agent to use for completion.
type AgentType string

const (
	AgentTypeLlama     AgentType = "llama"
	AgentTypeLocal     AgentType = "local"
	AgentTypeOpenAI    AgentType = "openai"
	AgentTypeAnthropic AgentType = "anthropic"
	AgentTypeGroq      AgentType = "groq"
	AgentTypeXAI       AgentType = "xai"
	AgentTypeOnline    AgentType = "online"
)

// CompleteTextRequest represents a text completion request.
type CompleteTextRequest struct {
	Prompt           string    `json:"prompt"`
	SystemPrompt     string    `json:"system_prompt,omitempty"`
	MaxTokens        int       `json:"max_tokens,omitempty"`
	N                int       `json:"n,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	Temperature      float64   `json:"temperature,omitempty"`
	TopP             float64   `json:"top_p,omitempty"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64   `json:"presence_penalty,omitempty"`
	Echo             bool      `json:"echo,omitempty"`
	BestOf           *int      `json:"best_of,omitempty"`
	PromptTokens     []int     `json:"prompt_tokens,omitempty"`
	ResponseFormat   string    `json:"response_format,omitempty"`
	AgentType        AgentType `json:"agent_type,omitempty"`
	Model            string    `json:"model,omitempty"`
}

// WithDefaults returns a copy of the request with default values applied.
func (r CompleteTextRequest) WithDefaults() CompleteTextRequest {
	if r.MaxTokens == 0 {
		r.MaxTokens = 256
	}
	if r.N == 0 {
		r.N = 1
	}
	if r.Temperature == 0 {
		r.Temperature = 0.7
	}
	if r.TopP == 0 {
		r.TopP = 1.0
	}
	if r.ResponseFormat == "" {
		r.ResponseFormat = "text"
	}
	if r.AgentType == "" {
		r.AgentType = AgentTypeLocal
	}
	return r
}

// CompleteTextResponse represents a text completion response.
type CompleteTextResponse struct {
	ID           string    `json:"id"`
	Object       string    `json:"object"`
	Created      int64     `json:"created"`
	Model        string    `json:"model"`
	Choices      []Choice  `json:"choices"`
	Usage        Usage     `json:"usage"`
	ChatID       int64     `json:"chat_id,omitempty"`
	SystemPrompt string    `json:"system_prompt,omitempty"`
}

// Choice represents a single completion choice.
type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Text         string   `json:"text,omitempty"`
	FinishReason string   `json:"finish_reason"`
	LogProbs     *LogProbs `json:"logprobs,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role       string          `json:"role"`
	Content    string          `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// ToolCall represents a function call made by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents the function being called.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// LogProbs represents token log probabilities.
type LogProbs struct {
	Tokens        []string             `json:"tokens,omitempty"`
	TokenLogProbs []float64            `json:"token_logprobs,omitempty"`
	TopLogProbs   []map[string]float64 `json:"top_logprobs,omitempty"`
	TextOffset    []int                `json:"text_offset,omitempty"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunk represents a single chunk in a streaming response.
type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// StreamChoice represents a choice in a streaming response.
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        *MessageDelta `json:"delta"`
	FinishReason string       `json:"finish_reason,omitempty"`
}

// MessageDelta represents incremental content in a streaming response.
type MessageDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// InitializeAgentRequest represents a request to initialize an agent.
type InitializeAgentRequest struct {
	Prompt    string    `json:"prompt"`
	AgentType AgentType `json:"agent_type,omitempty"`
	Model     string    `json:"model,omitempty"`
}

// InitializeAgentResponse represents the response from agent initialization.
type InitializeAgentResponse struct {
	ID        string `json:"id"`
	Status    string `json:"status"`
	Message   string `json:"message,omitempty"`
	SessionID string `json:"session_id"`
}

// TickRequest represents a request to advance the agent.
type TickRequest struct {
	SessionID string `json:"session_id"`
}

// TickResponse represents the response from an agent tick.
type TickResponse struct {
	ID       string   `json:"id"`
	Status   string   `json:"status"`
	Messages []string `json:"messages,omitempty"`
	Complete bool     `json:"complete"`
}

// ChatHistoryRequest represents a request to get chat history.
type ChatHistoryRequest struct {
	SessionID string `json:"session_id"`
	Limit     int    `json:"limit,omitempty"`
	Offset    int    `json:"offset,omitempty"`
}

// ChatHistoryResponse represents the chat history response.
type ChatHistoryResponse struct {
	SessionID string    `json:"session_id"`
	Messages  []Message `json:"messages"`
	Total     int       `json:"total"`
}

// ErrorResponse represents an error response.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error details.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
}

// NewErrorResponse creates a new error response.
func NewErrorResponse(message, errType string) ErrorResponse {
	return ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errType,
		},
	}
}

// NewCompleteTextResponse creates a new completion response.
func NewCompleteTextResponse(id, model, content, finishReason string, usage Usage) CompleteTextResponse {
	return CompleteTextResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Message: &Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}
}

// ToJSON converts the response to JSON bytes.
func (r CompleteTextResponse) ToJSON() ([]byte, error) {
	return json.Marshal(r)
}

// StreamChunkToSSE converts a stream chunk to Server-Sent Events format.
func StreamChunkToSSE(chunk StreamChunk) ([]byte, error) {
	data, err := json.Marshal(chunk)
	if err != nil {
		return nil, err
	}
	return append([]byte("data: "), append(data, '\n', '\n')...), nil
}
