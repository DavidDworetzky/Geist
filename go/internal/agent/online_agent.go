// Package agent provides the core agent interface and implementations.
package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"

	"github.com/DavidDworetzky/Geist/internal/models"
)

// OnlineAgent implements the Agent interface using HTTP APIs.
type OnlineAgent struct {
	client   *http.Client
	provider models.Provider
	model    string
	apiKey   string
	endpoint string
}

// NewOnlineAgent creates a new OnlineAgent.
func NewOnlineAgent(provider models.Provider, model, apiKey string) *OnlineAgent {
	return &OnlineAgent{
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
		provider: provider,
		model:    model,
		apiKey:   apiKey,
		endpoint: models.GetEndpoint(provider),
	}
}

// NewOnlineAgentWithEndpoint creates a new OnlineAgent with a custom endpoint (for testing).
func NewOnlineAgentWithEndpoint(provider models.Provider, model, apiKey, endpoint string) *OnlineAgent {
	return &OnlineAgent{
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
		provider: provider,
		model:    model,
		apiKey:   apiKey,
		endpoint: endpoint,
	}
}

// Type returns the agent type.
func (a *OnlineAgent) Type() models.AgentType {
	return models.AgentType(a.provider)
}

// Model returns the model being used.
func (a *OnlineAgent) Model() string {
	return a.model
}

// Close releases resources.
func (a *OnlineAgent) Close() error {
	return nil
}

// Initialize prepares the agent.
func (a *OnlineAgent) Initialize(ctx context.Context, taskPrompt string) error {
	return nil
}

// Tick advances the agent.
func (a *OnlineAgent) Tick(ctx context.Context) (*TickResult, error) {
	return &TickResult{Status: "idle", Complete: true}, nil
}

// CompleteText performs a text completion.
func (a *OnlineAgent) CompleteText(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	req = ptr(req.WithDefaults())

	switch a.provider {
	case models.ProviderAnthropic:
		return a.completeAnthropic(ctx, req)
	default:
		return a.completeOpenAICompatible(ctx, req)
	}
}

// StreamCompleteText performs a streaming text completion.
func (a *OnlineAgent) StreamCompleteText(ctx context.Context, req *CompletionRequest) (<-chan StreamToken, error) {
	req = ptr(req.WithDefaults())

	switch a.provider {
	case models.ProviderAnthropic:
		return a.streamAnthropic(ctx, req)
	default:
		return a.streamOpenAICompatible(ctx, req)
	}
}

// OpenAI-compatible completion (OpenAI, Groq, X.AI)
func (a *OnlineAgent) completeOpenAICompatible(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	messages := buildOpenAIMessages(req)

	body := map[string]interface{}{
		"model":             a.model,
		"messages":          messages,
		"max_tokens":        req.MaxTokens,
		"temperature":       req.Temperature,
		"top_p":             req.TopP,
		"frequency_penalty": req.FrequencyPenalty,
		"presence_penalty":  req.PresencePenalty,
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.endpoint, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+a.apiKey)

	resp, err := a.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &CompletionResponse{
		ID:           result.ID,
		Content:      result.Choices[0].Message.Content,
		FinishReason: result.Choices[0].FinishReason,
		Model:        a.model,
		Usage: Usage{
			PromptTokens:     result.Usage.PromptTokens,
			CompletionTokens: result.Usage.CompletionTokens,
			TotalTokens:      result.Usage.TotalTokens,
		},
	}, nil
}

// Anthropic completion
func (a *OnlineAgent) completeAnthropic(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	messages := buildAnthropicMessages(req)

	body := map[string]interface{}{
		"model":      a.model,
		"messages":   messages,
		"max_tokens": req.MaxTokens,
	}
	if req.SystemPrompt != "" {
		body["system"] = req.SystemPrompt
	}
	if req.Temperature > 0 {
		body["temperature"] = req.Temperature
	}
	if req.TopP > 0 && req.TopP < 1 {
		body["top_p"] = req.TopP
	}
	if len(req.Stop) > 0 {
		body["stop_sequences"] = req.Stop
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.endpoint, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := a.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		ID      string `json:"id"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	var content string
	for _, c := range result.Content {
		if c.Type == "text" {
			content += c.Text
		}
	}

	return &CompletionResponse{
		ID:           result.ID,
		Content:      content,
		FinishReason: result.StopReason,
		Model:        a.model,
		Usage: Usage{
			PromptTokens:     result.Usage.InputTokens,
			CompletionTokens: result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
		},
	}, nil
}

// OpenAI-compatible streaming
func (a *OnlineAgent) streamOpenAICompatible(ctx context.Context, req *CompletionRequest) (<-chan StreamToken, error) {
	messages := buildOpenAIMessages(req)

	body := map[string]interface{}{
		"model":             a.model,
		"messages":          messages,
		"max_tokens":        req.MaxTokens,
		"temperature":       req.Temperature,
		"top_p":             req.TopP,
		"frequency_penalty": req.FrequencyPenalty,
		"presence_penalty":  req.PresencePenalty,
		"stream":            true,
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.endpoint, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+a.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := a.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	tokens := make(chan StreamToken, 100)

	go func() {
		defer resp.Body.Close()
		defer close(tokens)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				tokens <- StreamToken{IsFinal: true, FinishReason: "stop"}
				return
			}

			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				log.Warn().Err(err).Str("data", data).Msg("Failed to parse stream chunk")
				continue
			}

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				if choice.Delta.Content != "" {
					tokens <- StreamToken{Text: choice.Delta.Content}
				}
				if choice.FinishReason != "" {
					tokens <- StreamToken{IsFinal: true, FinishReason: choice.FinishReason}
					return
				}
			}
		}

		if err := scanner.Err(); err != nil {
			tokens <- StreamToken{Error: err, IsFinal: true}
		}
	}()

	return tokens, nil
}

// Anthropic streaming
func (a *OnlineAgent) streamAnthropic(ctx context.Context, req *CompletionRequest) (<-chan StreamToken, error) {
	messages := buildAnthropicMessages(req)

	body := map[string]interface{}{
		"model":      a.model,
		"messages":   messages,
		"max_tokens": req.MaxTokens,
		"stream":     true,
	}
	if req.SystemPrompt != "" {
		body["system"] = req.SystemPrompt
	}
	if req.Temperature > 0 {
		body["temperature"] = req.Temperature
	}
	if len(req.Stop) > 0 {
		body["stop_sequences"] = req.Stop
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.endpoint, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := a.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	tokens := make(chan StreamToken, 100)

	go func() {
		defer resp.Body.Close()
		defer close(tokens)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")

			var event struct {
				Type  string `json:"type"`
				Delta struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"delta"`
			}

			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			switch event.Type {
			case "content_block_delta":
				if event.Delta.Text != "" {
					tokens <- StreamToken{Text: event.Delta.Text}
				}
			case "message_stop":
				tokens <- StreamToken{IsFinal: true, FinishReason: "stop"}
				return
			}
		}

		if err := scanner.Err(); err != nil {
			tokens <- StreamToken{Error: err, IsFinal: true}
		}
	}()

	return tokens, nil
}

// Helper functions

func buildOpenAIMessages(req *CompletionRequest) []map[string]string {
	var messages []map[string]string

	if req.SystemPrompt != "" {
		messages = append(messages, map[string]string{
			"role":    "system",
			"content": req.SystemPrompt,
		})
	}

	for _, msg := range req.ChatHistory {
		messages = append(messages, map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	messages = append(messages, map[string]string{
		"role":    "user",
		"content": req.Prompt,
	})

	return messages
}

func buildAnthropicMessages(req *CompletionRequest) []map[string]string {
	var messages []map[string]string

	for _, msg := range req.ChatHistory {
		if msg.Role != "system" {
			messages = append(messages, map[string]string{
				"role":    msg.Role,
				"content": msg.Content,
			})
		}
	}

	messages = append(messages, map[string]string{
		"role":    "user",
		"content": req.Prompt,
	})

	return messages
}

func ptr[T any](v T) *T {
	return &v
}

// GenerateID generates a unique completion ID.
func GenerateID() string {
	return "chatcmpl-" + uuid.New().String()[:8]
}
