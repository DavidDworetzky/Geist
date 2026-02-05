// Package handlers provides HTTP request handlers for the API.
package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/DavidDworetzky/Geist/internal/database/repository"
	"github.com/DavidDworetzky/Geist/internal/models"
)

// UserSettingsHandler handles user settings HTTP requests.
type UserSettingsHandler struct {
	repo *repository.UserSettingsRepository
}

// NewUserSettingsHandler creates a new UserSettingsHandler.
func NewUserSettingsHandler(repo *repository.UserSettingsRepository) *UserSettingsHandler {
	return &UserSettingsHandler{repo: repo}
}

// GetSettings handles GET /api/v1/user-settings
func (h *UserSettingsHandler) GetSettings(w http.ResponseWriter, r *http.Request) {
	userID := int64(1) // TODO: Get from auth context

	settings, err := h.repo.GetByUserID(r.Context(), userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to get settings", err.Error())
		return
	}

	if settings == nil {
		// Return default settings
		settings = ptr(models.DefaultUserSettings(userID))
	}

	writeJSON(w, http.StatusOK, settings)
}

// UpdateSettings handles PUT /api/v1/user-settings
func (h *UserSettingsHandler) UpdateSettings(w http.ResponseWriter, r *http.Request) {
	userID := int64(1) // TODO: Get from auth context

	var req models.UserSettings
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	req.UserID = userID

	if err := h.repo.Upsert(r.Context(), &req); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to update settings", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, req)
}

// PatchSettings handles PATCH /api/v1/user-settings
func (h *UserSettingsHandler) PatchSettings(w http.ResponseWriter, r *http.Request) {
	userID := int64(1) // TODO: Get from auth context

	// Get existing settings
	existing, err := h.repo.GetByUserID(r.Context(), userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to get settings", err.Error())
		return
	}

	if existing == nil {
		existing = ptr(models.DefaultUserSettings(userID))
	}

	// Decode patch
	var patch map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&patch); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	// Apply patch
	if v, ok := patch["default_agent_type"].(string); ok {
		existing.DefaultAgentType = models.AgentType(v)
	}
	if v, ok := patch["default_model"].(string); ok {
		existing.DefaultModel = v
	}
	if v, ok := patch["default_system_prompt"].(string); ok {
		existing.DefaultSystemPrompt = v
	}
	if v, ok := patch["preferences"].(map[string]interface{}); ok {
		if theme, ok := v["theme"].(string); ok {
			existing.Preferences.Theme = theme
		}
		if lang, ok := v["language"].(string); ok {
			existing.Preferences.Language = lang
		}
		if stream, ok := v["stream_responses"].(bool); ok {
			existing.Preferences.StreamResponses = stream
		}
		if save, ok := v["save_chat_history"].(bool); ok {
			existing.Preferences.SaveChatHistory = save
		}
		if maxLen, ok := v["max_history_length"].(float64); ok {
			existing.Preferences.MaxHistoryLength = int(maxLen)
		}
	}

	if err := h.repo.Upsert(r.Context(), existing); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to update settings", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, existing)
}

// ResetSettings handles DELETE /api/v1/user-settings
func (h *UserSettingsHandler) ResetSettings(w http.ResponseWriter, r *http.Request) {
	userID := int64(1) // TODO: Get from auth context

	defaults := models.DefaultUserSettings(userID)

	if err := h.repo.Upsert(r.Context(), &defaults); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to reset settings", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, defaults)
}

func ptr[T any](v T) *T {
	return &v
}
