// Package models provides data models for API requests and responses.
package models

import (
	"encoding/json"
	"time"
)

// AgentSettings contains configuration for an agent instance.
type AgentSettings struct {
	Model            string            `json:"model"`
	SystemPrompt     string            `json:"system_prompt"`
	MaxTokens        int               `json:"max_tokens"`
	Temperature      float64           `json:"temperature"`
	TopP             float64           `json:"top_p"`
	FrequencyPenalty float64           `json:"frequency_penalty"`
	PresencePenalty  float64           `json:"presence_penalty"`
	Stop             []string          `json:"stop,omitempty"`
	Extra            map[string]string `json:"extra,omitempty"`
}

// DefaultAgentSettings returns settings with sensible defaults.
func DefaultAgentSettings() AgentSettings {
	return AgentSettings{
		Model:            "meta-llama/Llama-3.1-8B-Instruct",
		SystemPrompt:     "You are a helpful AI assistant.",
		MaxTokens:        2048,
		Temperature:      0.7,
		TopP:             1.0,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
	}
}

// Merge merges override settings into the current settings.
func (s AgentSettings) Merge(override AgentSettings) AgentSettings {
	result := s
	if override.Model != "" {
		result.Model = override.Model
	}
	if override.SystemPrompt != "" {
		result.SystemPrompt = override.SystemPrompt
	}
	if override.MaxTokens > 0 {
		result.MaxTokens = override.MaxTokens
	}
	if override.Temperature > 0 {
		result.Temperature = override.Temperature
	}
	if override.TopP > 0 {
		result.TopP = override.TopP
	}
	if override.FrequencyPenalty != 0 {
		result.FrequencyPenalty = override.FrequencyPenalty
	}
	if override.PresencePenalty != 0 {
		result.PresencePenalty = override.PresencePenalty
	}
	if len(override.Stop) > 0 {
		result.Stop = override.Stop
	}
	return result
}

// AgentContext holds the runtime context for an agent.
type AgentContext struct {
	ID               string            `json:"id"`
	Settings         AgentSettings     `json:"settings"`
	WorldContext     []string          `json:"world_context"`
	TaskContext      []string          `json:"task_context"`
	ExecutionContext []string          `json:"execution_context"`
	Envs             map[string]string `json:"envs"`
	CreatedAt        time.Time         `json:"created_at"`
}

// NewAgentContext creates a new agent context with the given settings.
func NewAgentContext(id string, settings AgentSettings) *AgentContext {
	return &AgentContext{
		ID:               id,
		Settings:         settings,
		WorldContext:     make([]string, 0),
		TaskContext:      make([]string, 0),
		ExecutionContext: make([]string, 0),
		Envs:             make(map[string]string),
		CreatedAt:        time.Now(),
	}
}

// AddWorldContext adds context about the world/environment.
func (c *AgentContext) AddWorldContext(ctx string) {
	c.WorldContext = append(c.WorldContext, ctx)
}

// AddTaskContext adds context about the current task.
func (c *AgentContext) AddTaskContext(ctx string) {
	c.TaskContext = append(c.TaskContext, ctx)
}

// AddExecutionContext adds context about execution state.
func (c *AgentContext) AddExecutionContext(ctx string) {
	c.ExecutionContext = append(c.ExecutionContext, ctx)
}

// AgentPreset represents a saved agent configuration.
type AgentPreset struct {
	ID           int64          `json:"id"`
	Name         string         `json:"name"`
	Description  string         `json:"description,omitempty"`
	AgentType    AgentType      `json:"agent_type"`
	Settings     AgentSettings  `json:"settings"`
	CreatedAt    time.Time      `json:"created_at"`
	UpdatedAt    time.Time      `json:"updated_at"`
}

// UserSettings represents user-level settings.
type UserSettings struct {
	ID                  int64         `json:"id"`
	UserID              int64         `json:"user_id"`
	DefaultAgentType    AgentType     `json:"default_agent_type"`
	DefaultModel        string        `json:"default_model"`
	DefaultSystemPrompt string        `json:"default_system_prompt"`
	Preferences         Preferences   `json:"preferences"`
	CreatedAt           time.Time     `json:"created_at"`
	UpdatedAt           time.Time     `json:"updated_at"`
}

// Preferences holds user preferences.
type Preferences struct {
	Theme            string `json:"theme,omitempty"`
	Language         string `json:"language,omitempty"`
	StreamResponses  bool   `json:"stream_responses"`
	SaveChatHistory  bool   `json:"save_chat_history"`
	MaxHistoryLength int    `json:"max_history_length,omitempty"`
}

// DefaultUserSettings returns settings with sensible defaults.
func DefaultUserSettings(userID int64) UserSettings {
	return UserSettings{
		UserID:              userID,
		DefaultAgentType:    AgentTypeLocal,
		DefaultModel:        "meta-llama/Llama-3.1-8B-Instruct",
		DefaultSystemPrompt: "You are a helpful AI assistant.",
		Preferences: Preferences{
			Theme:            "system",
			Language:         "en",
			StreamResponses:  true,
			SaveChatHistory:  true,
			MaxHistoryLength: 100,
		},
	}
}

// ChatSession represents a chat session.
type ChatSession struct {
	ID        int64     `json:"id"`
	UserID    int64     `json:"user_id"`
	AgentType AgentType `json:"agent_type"`
	Model     string    `json:"model"`
	Title     string    `json:"title,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// ChatMessage represents a single message in a chat session.
type ChatMessage struct {
	ID        int64     `json:"id"`
	SessionID int64     `json:"session_id"`
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Metadata  string    `json:"metadata,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// ToMessage converts ChatMessage to Message.
func (m ChatMessage) ToMessage() Message {
	return Message{
		Role:    m.Role,
		Content: m.Content,
	}
}

// MetadataJSON parses the metadata as JSON.
func (m ChatMessage) MetadataJSON() (map[string]interface{}, error) {
	if m.Metadata == "" {
		return nil, nil
	}
	var result map[string]interface{}
	err := json.Unmarshal([]byte(m.Metadata), &result)
	return result, err
}

// VoiceSession represents a voice interaction session.
type VoiceSession struct {
	ID            string    `json:"id"`
	ChatSessionID int64     `json:"chat_session_id,omitempty"`
	Status        string    `json:"status"`
	STTProvider   string    `json:"stt_provider"`
	TTSProvider   string    `json:"tts_provider"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// FileUpload represents an uploaded file.
type FileUpload struct {
	ID          int64     `json:"id"`
	UserID      int64     `json:"user_id"`
	Filename    string    `json:"filename"`
	ContentType string    `json:"content_type"`
	Size        int64     `json:"size"`
	StoragePath string    `json:"storage_path"`
	Checksum    string    `json:"checksum,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

// Workflow represents an automated workflow.
type Workflow struct {
	ID          int64           `json:"id"`
	UserID      int64           `json:"user_id"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Definition  json.RawMessage `json:"definition"`
	Enabled     bool            `json:"enabled"`
	CreatedAt   time.Time       `json:"created_at"`
	UpdatedAt   time.Time       `json:"updated_at"`
}

// WorkflowExecution represents a single execution of a workflow.
type WorkflowExecution struct {
	ID         int64           `json:"id"`
	WorkflowID int64           `json:"workflow_id"`
	Status     string          `json:"status"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     json.RawMessage `json:"output,omitempty"`
	Error      string          `json:"error,omitempty"`
	StartedAt  time.Time       `json:"started_at"`
	FinishedAt *time.Time      `json:"finished_at,omitempty"`
}
