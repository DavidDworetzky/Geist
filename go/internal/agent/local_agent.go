// Package agent provides the core agent interface and implementations.
package agent

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	"github.com/DavidDworetzky/Geist/internal/models"
	pb "github.com/DavidDworetzky/Geist/internal/inference/proto"
)

// LocalAgent implements the Agent interface using local MLX inference via gRPC.
type LocalAgent struct {
	conn   *grpc.ClientConn
	client pb.InferenceServiceClient
	model  string
}

// NewLocalAgent creates a new LocalAgent connected to the inference service.
func NewLocalAgent(address, model string) (*LocalAgent, error) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	}

	conn, err := grpc.Dial(address, opts...)
	if err != nil {
		return nil, fmt.Errorf("connecting to inference service: %w", err)
	}

	return &LocalAgent{
		conn:   conn,
		client: pb.NewInferenceServiceClient(conn),
		model:  model,
	}, nil
}

// Type returns the agent type.
func (a *LocalAgent) Type() models.AgentType {
	return models.AgentTypeLocal
}

// Model returns the model being used.
func (a *LocalAgent) Model() string {
	return a.model
}

// Close releases resources.
func (a *LocalAgent) Close() error {
	if a.conn != nil {
		return a.conn.Close()
	}
	return nil
}

// Initialize prepares the agent.
func (a *LocalAgent) Initialize(ctx context.Context, taskPrompt string) error {
	return nil
}

// Tick advances the agent.
func (a *LocalAgent) Tick(ctx context.Context) (*TickResult, error) {
	return &TickResult{Status: "idle", Complete: true}, nil
}

// CompleteText performs a text completion using the local inference service.
func (a *LocalAgent) CompleteText(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	req = ptr(req.WithDefaults())

	// Build chat history
	var history []*pb.Message
	for _, msg := range req.ChatHistory {
		history = append(history, &pb.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	grpcReq := &pb.CompletionRequest{
		ModelId:      a.model,
		SystemPrompt: req.SystemPrompt,
		UserPrompt:   req.Prompt,
		Config: &pb.GenerationConfig{
			MaxTokens:        int32(req.MaxTokens),
			Temperature:      float32(req.Temperature),
			TopP:             float32(req.TopP),
			FrequencyPenalty: float32(req.FrequencyPenalty),
			PresencePenalty:  float32(req.PresencePenalty),
			StopSequences:    req.Stop,
		},
		ChatHistory: history,
	}

	resp, err := a.client.Complete(ctx, grpcReq)
	if err != nil {
		return nil, fmt.Errorf("inference request failed: %w", err)
	}

	return &CompletionResponse{
		ID:           resp.Id,
		Content:      resp.Content,
		FinishReason: resp.FinishReason,
		Model:        resp.ModelId,
		Usage: Usage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		},
	}, nil
}

// StreamCompleteText performs a streaming text completion.
func (a *LocalAgent) StreamCompleteText(ctx context.Context, req *CompletionRequest) (<-chan StreamToken, error) {
	req = ptr(req.WithDefaults())

	// Build chat history
	var history []*pb.Message
	for _, msg := range req.ChatHistory {
		history = append(history, &pb.Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	grpcReq := &pb.CompletionRequest{
		ModelId:      a.model,
		SystemPrompt: req.SystemPrompt,
		UserPrompt:   req.Prompt,
		Config: &pb.GenerationConfig{
			MaxTokens:        int32(req.MaxTokens),
			Temperature:      float32(req.Temperature),
			TopP:             float32(req.TopP),
			FrequencyPenalty: float32(req.FrequencyPenalty),
			PresencePenalty:  float32(req.PresencePenalty),
			StopSequences:    req.Stop,
		},
		ChatHistory: history,
	}

	stream, err := a.client.StreamComplete(ctx, grpcReq)
	if err != nil {
		return nil, fmt.Errorf("stream request failed: %w", err)
	}

	tokens := make(chan StreamToken, 100)

	go func() {
		defer close(tokens)

		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				log.Error().Err(err).Msg("Stream error")
				tokens <- StreamToken{Error: err, IsFinal: true}
				return
			}

			tokens <- StreamToken{
				Text:         chunk.Token,
				IsFinal:      chunk.IsFinal,
				FinishReason: chunk.FinishReason,
			}

			if chunk.IsFinal {
				return
			}
		}
	}()

	return tokens, nil
}

// LoadModel loads a model into the inference service.
func (a *LocalAgent) LoadModel(ctx context.Context, modelID, runnerType string) error {
	req := &pb.LoadModelRequest{
		ModelId:    modelID,
		RunnerType: runnerType,
	}

	resp, err := a.client.LoadModel(ctx, req)
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}

	if !resp.Success {
		return fmt.Errorf("failed to load model: %s", resp.Message)
	}

	log.Info().
		Str("model", modelID).
		Int64("load_time_ms", resp.LoadTimeMs).
		Int64("memory_bytes", resp.MemoryUsageBytes).
		Msg("Model loaded")

	return nil
}

// UnloadModel unloads a model from the inference service.
func (a *LocalAgent) UnloadModel(ctx context.Context, modelID string) error {
	req := &pb.UnloadModelRequest{
		ModelId: modelID,
	}

	resp, err := a.client.UnloadModel(ctx, req)
	if err != nil {
		return fmt.Errorf("unloading model: %w", err)
	}

	if !resp.Success {
		return fmt.Errorf("failed to unload model: %s", resp.Message)
	}

	log.Info().Str("model", modelID).Msg("Model unloaded")
	return nil
}

// ListModels lists available models in the inference service.
func (a *LocalAgent) ListModels(ctx context.Context) ([]*pb.ModelInfo, error) {
	req := &pb.ListModelsRequest{
		IncludeUnloaded: true,
	}

	resp, err := a.client.ListModels(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("listing models: %w", err)
	}

	return resp.Models, nil
}

// Health checks the health of the inference service.
func (a *LocalAgent) Health(ctx context.Context) (*pb.HealthResponse, error) {
	resp, err := a.client.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	return resp, nil
}

// InferencePool manages a pool of connections to the inference service.
type InferencePool struct {
	agents  []*LocalAgent
	current int
	mu      chan struct{} // Simple mutex using channel
}

// NewInferencePool creates a new connection pool.
func NewInferencePool(address, model string, size int) (*InferencePool, error) {
	agents := make([]*LocalAgent, size)
	for i := 0; i < size; i++ {
		agent, err := NewLocalAgent(address, model)
		if err != nil {
			// Close any agents we've already created
			for j := 0; j < i; j++ {
				agents[j].Close()
			}
			return nil, fmt.Errorf("creating agent %d: %w", i, err)
		}
		agents[i] = agent
	}

	return &InferencePool{
		agents:  agents,
		current: 0,
		mu:      make(chan struct{}, 1),
	}, nil
}

// Get returns an agent from the pool using round-robin.
func (p *InferencePool) Get() *LocalAgent {
	p.mu <- struct{}{}
	defer func() { <-p.mu }()

	agent := p.agents[p.current]
	p.current = (p.current + 1) % len(p.agents)
	return agent
}

// Close closes all agents in the pool.
func (p *InferencePool) Close() error {
	for _, agent := range p.agents {
		if err := agent.Close(); err != nil {
			log.Error().Err(err).Msg("Error closing agent")
		}
	}
	return nil
}
