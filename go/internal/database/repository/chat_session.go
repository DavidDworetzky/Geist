// Package repository provides data access for domain entities.
package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/DavidDworetzky/Geist/internal/models"
)

// ChatSessionRepository provides data access for chat sessions.
type ChatSessionRepository struct {
	pool *pgxpool.Pool
}

// NewChatSessionRepository creates a new ChatSessionRepository.
func NewChatSessionRepository(pool *pgxpool.Pool) *ChatSessionRepository {
	return &ChatSessionRepository{pool: pool}
}

// Create creates a new chat session.
func (r *ChatSessionRepository) Create(ctx context.Context, session *models.ChatSession) error {
	query := `
		INSERT INTO chat_sessions (user_id, agent_type, model, title, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id`

	now := time.Now()
	session.CreatedAt = now
	session.UpdatedAt = now

	err := r.pool.QueryRow(ctx, query,
		session.UserID,
		session.AgentType,
		session.Model,
		session.Title,
		session.CreatedAt,
		session.UpdatedAt,
	).Scan(&session.ID)

	if err != nil {
		return fmt.Errorf("creating chat session: %w", err)
	}
	return nil
}

// GetByID retrieves a chat session by ID.
func (r *ChatSessionRepository) GetByID(ctx context.Context, id int64) (*models.ChatSession, error) {
	query := `
		SELECT id, user_id, agent_type, model, title, created_at, updated_at
		FROM chat_sessions
		WHERE id = $1`

	session := &models.ChatSession{}
	err := r.pool.QueryRow(ctx, query, id).Scan(
		&session.ID,
		&session.UserID,
		&session.AgentType,
		&session.Model,
		&session.Title,
		&session.CreatedAt,
		&session.UpdatedAt,
	)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("getting chat session: %w", err)
	}
	return session, nil
}

// GetByUserID retrieves chat sessions for a user.
func (r *ChatSessionRepository) GetByUserID(ctx context.Context, userID int64, limit, offset int) ([]*models.ChatSession, error) {
	query := `
		SELECT id, user_id, agent_type, model, title, created_at, updated_at
		FROM chat_sessions
		WHERE user_id = $1
		ORDER BY updated_at DESC
		LIMIT $2 OFFSET $3`

	rows, err := r.pool.Query(ctx, query, userID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("listing chat sessions: %w", err)
	}
	defer rows.Close()

	var sessions []*models.ChatSession
	for rows.Next() {
		session := &models.ChatSession{}
		if err := rows.Scan(
			&session.ID,
			&session.UserID,
			&session.AgentType,
			&session.Model,
			&session.Title,
			&session.CreatedAt,
			&session.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scanning chat session: %w", err)
		}
		sessions = append(sessions, session)
	}
	return sessions, nil
}

// Update updates a chat session.
func (r *ChatSessionRepository) Update(ctx context.Context, session *models.ChatSession) error {
	query := `
		UPDATE chat_sessions
		SET agent_type = $2, model = $3, title = $4, updated_at = $5
		WHERE id = $1`

	session.UpdatedAt = time.Now()
	_, err := r.pool.Exec(ctx, query,
		session.ID,
		session.AgentType,
		session.Model,
		session.Title,
		session.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("updating chat session: %w", err)
	}
	return nil
}

// Delete deletes a chat session.
func (r *ChatSessionRepository) Delete(ctx context.Context, id int64) error {
	_, err := r.pool.Exec(ctx, "DELETE FROM chat_sessions WHERE id = $1", id)
	if err != nil {
		return fmt.Errorf("deleting chat session: %w", err)
	}
	return nil
}

// AddMessage adds a message to a chat session.
func (r *ChatSessionRepository) AddMessage(ctx context.Context, msg *models.ChatMessage) error {
	query := `
		INSERT INTO chat_messages (session_id, role, content, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id`

	msg.CreatedAt = time.Now()

	var metadata interface{}
	if msg.Metadata != "" {
		metadata = msg.Metadata
	}

	err := r.pool.QueryRow(ctx, query,
		msg.SessionID,
		msg.Role,
		msg.Content,
		metadata,
		msg.CreatedAt,
	).Scan(&msg.ID)

	if err != nil {
		return fmt.Errorf("adding message: %w", err)
	}

	// Update session timestamp
	_, err = r.pool.Exec(ctx,
		"UPDATE chat_sessions SET updated_at = $1 WHERE id = $2",
		msg.CreatedAt, msg.SessionID)
	if err != nil {
		return fmt.Errorf("updating session timestamp: %w", err)
	}

	return nil
}

// GetMessages retrieves messages for a chat session.
func (r *ChatSessionRepository) GetMessages(ctx context.Context, sessionID int64, limit, offset int) ([]*models.ChatMessage, error) {
	query := `
		SELECT id, session_id, role, content, COALESCE(metadata::text, ''), created_at
		FROM chat_messages
		WHERE session_id = $1
		ORDER BY created_at ASC
		LIMIT $2 OFFSET $3`

	rows, err := r.pool.Query(ctx, query, sessionID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("getting messages: %w", err)
	}
	defer rows.Close()

	var messages []*models.ChatMessage
	for rows.Next() {
		msg := &models.ChatMessage{}
		if err := rows.Scan(
			&msg.ID,
			&msg.SessionID,
			&msg.Role,
			&msg.Content,
			&msg.Metadata,
			&msg.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scanning message: %w", err)
		}
		messages = append(messages, msg)
	}
	return messages, nil
}

// GetMessageCount returns the number of messages in a session.
func (r *ChatSessionRepository) GetMessageCount(ctx context.Context, sessionID int64) (int, error) {
	var count int
	err := r.pool.QueryRow(ctx,
		"SELECT COUNT(*) FROM chat_messages WHERE session_id = $1",
		sessionID,
	).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("counting messages: %w", err)
	}
	return count, nil
}

// DeleteMessages deletes all messages in a session.
func (r *ChatSessionRepository) DeleteMessages(ctx context.Context, sessionID int64) error {
	_, err := r.pool.Exec(ctx, "DELETE FROM chat_messages WHERE session_id = $1", sessionID)
	if err != nil {
		return fmt.Errorf("deleting messages: %w", err)
	}
	return nil
}

// AgentPresetRepository provides data access for agent presets.
type AgentPresetRepository struct {
	pool *pgxpool.Pool
}

// NewAgentPresetRepository creates a new AgentPresetRepository.
func NewAgentPresetRepository(pool *pgxpool.Pool) *AgentPresetRepository {
	return &AgentPresetRepository{pool: pool}
}

// Create creates a new agent preset.
func (r *AgentPresetRepository) Create(ctx context.Context, preset *models.AgentPreset) error {
	query := `
		INSERT INTO agent_presets (name, description, agent_type, settings, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id`

	now := time.Now()
	preset.CreatedAt = now
	preset.UpdatedAt = now

	settingsJSON, err := json.Marshal(preset.Settings)
	if err != nil {
		return fmt.Errorf("marshaling settings: %w", err)
	}

	err = r.pool.QueryRow(ctx, query,
		preset.Name,
		preset.Description,
		preset.AgentType,
		settingsJSON,
		preset.CreatedAt,
		preset.UpdatedAt,
	).Scan(&preset.ID)

	if err != nil {
		return fmt.Errorf("creating agent preset: %w", err)
	}
	return nil
}

// GetByID retrieves an agent preset by ID.
func (r *AgentPresetRepository) GetByID(ctx context.Context, id int64) (*models.AgentPreset, error) {
	query := `
		SELECT id, name, description, agent_type, settings, created_at, updated_at
		FROM agent_presets
		WHERE id = $1`

	preset := &models.AgentPreset{}
	var settingsJSON []byte
	err := r.pool.QueryRow(ctx, query, id).Scan(
		&preset.ID,
		&preset.Name,
		&preset.Description,
		&preset.AgentType,
		&settingsJSON,
		&preset.CreatedAt,
		&preset.UpdatedAt,
	)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("getting agent preset: %w", err)
	}

	if err := json.Unmarshal(settingsJSON, &preset.Settings); err != nil {
		return nil, fmt.Errorf("unmarshaling settings: %w", err)
	}

	return preset, nil
}

// GetByName retrieves an agent preset by name.
func (r *AgentPresetRepository) GetByName(ctx context.Context, name string) (*models.AgentPreset, error) {
	query := `
		SELECT id, name, description, agent_type, settings, created_at, updated_at
		FROM agent_presets
		WHERE name = $1`

	preset := &models.AgentPreset{}
	var settingsJSON []byte
	err := r.pool.QueryRow(ctx, query, name).Scan(
		&preset.ID,
		&preset.Name,
		&preset.Description,
		&preset.AgentType,
		&settingsJSON,
		&preset.CreatedAt,
		&preset.UpdatedAt,
	)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("getting agent preset: %w", err)
	}

	if err := json.Unmarshal(settingsJSON, &preset.Settings); err != nil {
		return nil, fmt.Errorf("unmarshaling settings: %w", err)
	}

	return preset, nil
}

// List retrieves all agent presets.
func (r *AgentPresetRepository) List(ctx context.Context) ([]*models.AgentPreset, error) {
	query := `
		SELECT id, name, description, agent_type, settings, created_at, updated_at
		FROM agent_presets
		ORDER BY name ASC`

	rows, err := r.pool.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("listing agent presets: %w", err)
	}
	defer rows.Close()

	var presets []*models.AgentPreset
	for rows.Next() {
		preset := &models.AgentPreset{}
		var settingsJSON []byte
		if err := rows.Scan(
			&preset.ID,
			&preset.Name,
			&preset.Description,
			&preset.AgentType,
			&settingsJSON,
			&preset.CreatedAt,
			&preset.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scanning agent preset: %w", err)
		}
		if err := json.Unmarshal(settingsJSON, &preset.Settings); err != nil {
			return nil, fmt.Errorf("unmarshaling settings: %w", err)
		}
		presets = append(presets, preset)
	}
	return presets, nil
}

// Update updates an agent preset.
func (r *AgentPresetRepository) Update(ctx context.Context, preset *models.AgentPreset) error {
	query := `
		UPDATE agent_presets
		SET name = $2, description = $3, agent_type = $4, settings = $5, updated_at = $6
		WHERE id = $1`

	preset.UpdatedAt = time.Now()

	settingsJSON, err := json.Marshal(preset.Settings)
	if err != nil {
		return fmt.Errorf("marshaling settings: %w", err)
	}

	_, err = r.pool.Exec(ctx, query,
		preset.ID,
		preset.Name,
		preset.Description,
		preset.AgentType,
		settingsJSON,
		preset.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("updating agent preset: %w", err)
	}
	return nil
}

// Delete deletes an agent preset.
func (r *AgentPresetRepository) Delete(ctx context.Context, id int64) error {
	_, err := r.pool.Exec(ctx, "DELETE FROM agent_presets WHERE id = $1", id)
	if err != nil {
		return fmt.Errorf("deleting agent preset: %w", err)
	}
	return nil
}

// UserSettingsRepository provides data access for user settings.
type UserSettingsRepository struct {
	pool *pgxpool.Pool
}

// NewUserSettingsRepository creates a new UserSettingsRepository.
func NewUserSettingsRepository(pool *pgxpool.Pool) *UserSettingsRepository {
	return &UserSettingsRepository{pool: pool}
}

// GetByUserID retrieves user settings by user ID.
func (r *UserSettingsRepository) GetByUserID(ctx context.Context, userID int64) (*models.UserSettings, error) {
	query := `
		SELECT id, user_id, default_agent_type, default_model, default_system_prompt, preferences, created_at, updated_at
		FROM user_settings
		WHERE user_id = $1`

	settings := &models.UserSettings{}
	var preferencesJSON []byte
	err := r.pool.QueryRow(ctx, query, userID).Scan(
		&settings.ID,
		&settings.UserID,
		&settings.DefaultAgentType,
		&settings.DefaultModel,
		&settings.DefaultSystemPrompt,
		&preferencesJSON,
		&settings.CreatedAt,
		&settings.UpdatedAt,
	)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("getting user settings: %w", err)
	}

	if preferencesJSON != nil {
		if err := json.Unmarshal(preferencesJSON, &settings.Preferences); err != nil {
			return nil, fmt.Errorf("unmarshaling preferences: %w", err)
		}
	}

	return settings, nil
}

// Upsert creates or updates user settings.
func (r *UserSettingsRepository) Upsert(ctx context.Context, settings *models.UserSettings) error {
	query := `
		INSERT INTO user_settings (user_id, default_agent_type, default_model, default_system_prompt, preferences, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (user_id) DO UPDATE SET
			default_agent_type = EXCLUDED.default_agent_type,
			default_model = EXCLUDED.default_model,
			default_system_prompt = EXCLUDED.default_system_prompt,
			preferences = EXCLUDED.preferences,
			updated_at = EXCLUDED.updated_at
		RETURNING id`

	now := time.Now()
	if settings.CreatedAt.IsZero() {
		settings.CreatedAt = now
	}
	settings.UpdatedAt = now

	preferencesJSON, err := json.Marshal(settings.Preferences)
	if err != nil {
		return fmt.Errorf("marshaling preferences: %w", err)
	}

	err = r.pool.QueryRow(ctx, query,
		settings.UserID,
		settings.DefaultAgentType,
		settings.DefaultModel,
		settings.DefaultSystemPrompt,
		preferencesJSON,
		settings.CreatedAt,
		settings.UpdatedAt,
	).Scan(&settings.ID)

	if err != nil {
		return fmt.Errorf("upserting user settings: %w", err)
	}
	return nil
}
