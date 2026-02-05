// Package handlers provides HTTP request handlers for the API.
package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog/log"

	"github.com/DavidDworetzky/Geist/internal/agent"
	"github.com/DavidDworetzky/Geist/internal/database/repository"
	"github.com/DavidDworetzky/Geist/internal/models"
)

// AgentHandler handles agent-related HTTP requests.
type AgentHandler struct {
	factory     *agent.Factory
	chatRepo    *repository.ChatSessionRepository
}

// NewAgentHandler creates a new AgentHandler.
func NewAgentHandler(factory *agent.Factory, chatRepo *repository.ChatSessionRepository) *AgentHandler {
	return &AgentHandler{
		factory:  factory,
		chatRepo: chatRepo,
	}
}

// CompleteText handles POST /agent/complete_text
func (h *AgentHandler) CompleteText(w http.ResponseWriter, r *http.Request) {
	var req models.CompleteTextRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	req = req.WithDefaults()

	// Get agent
	ag, err := h.factory.AgentFromRequest(&req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to create agent", err.Error())
		return
	}

	// Build completion request
	completionReq := &agent.CompletionRequest{
		Prompt:           req.Prompt,
		SystemPrompt:     req.SystemPrompt,
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		FrequencyPenalty: req.FrequencyPenalty,
		PresencePenalty:  req.PresencePenalty,
		Stop:             req.Stop,
	}

	// Execute completion
	resp, err := ag.CompleteText(r.Context(), completionReq)
	if err != nil {
		log.Error().Err(err).Msg("Completion failed")
		writeError(w, http.StatusInternalServerError, "Completion failed", err.Error())
		return
	}

	// Build response
	apiResp := models.NewCompleteTextResponse(
		resp.ID,
		resp.Model,
		resp.Content,
		resp.FinishReason,
		models.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	)

	writeJSON(w, http.StatusOK, apiResp)
}

// StreamCompleteText handles POST /agent/stream_complete_text
func (h *AgentHandler) StreamCompleteText(w http.ResponseWriter, r *http.Request) {
	var req models.CompleteTextRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	req = req.WithDefaults()

	// Get agent
	ag, err := h.factory.AgentFromRequest(&req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to create agent", err.Error())
		return
	}

	// Build completion request
	completionReq := &agent.CompletionRequest{
		Prompt:           req.Prompt,
		SystemPrompt:     req.SystemPrompt,
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		FrequencyPenalty: req.FrequencyPenalty,
		PresencePenalty:  req.PresencePenalty,
		Stop:             req.Stop,
	}

	// Set up SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "Streaming not supported", "")
		return
	}

	// Start streaming
	tokens, err := ag.StreamCompleteText(r.Context(), completionReq)
	if err != nil {
		log.Error().Err(err).Msg("Stream failed")
		writeSSEError(w, flusher, err.Error())
		return
	}

	id := agent.GenerateID()
	for token := range tokens {
		if token.Error != nil {
			writeSSEError(w, flusher, token.Error.Error())
			return
		}

		chunk := models.StreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   ag.Model(),
			Choices: []models.StreamChoice{
				{
					Index: 0,
					Delta: &models.MessageDelta{Content: token.Text},
				},
			},
		}

		if token.IsFinal {
			chunk.Choices[0].FinishReason = token.FinishReason
		}

		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		if token.IsFinal {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
	}
}

// CompleteTextWithSession handles POST /agent/complete_text/{session_id}
func (h *AgentHandler) CompleteTextWithSession(w http.ResponseWriter, r *http.Request) {
	sessionIDStr := chi.URLParam(r, "session_id")
	sessionID, err := strconv.ParseInt(sessionIDStr, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "Invalid session ID", err.Error())
		return
	}

	var req models.CompleteTextRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	req = req.WithDefaults()

	// Get session
	session, err := h.chatRepo.GetByID(r.Context(), sessionID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to get session", err.Error())
		return
	}
	if session == nil {
		writeError(w, http.StatusNotFound, "Session not found", "")
		return
	}

	// Get chat history
	messages, err := h.chatRepo.GetMessages(r.Context(), sessionID, 100, 0)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to get chat history", err.Error())
		return
	}

	// Convert to models.Message
	var chatHistory []models.Message
	for _, msg := range messages {
		chatHistory = append(chatHistory, msg.ToMessage())
	}

	// Get agent
	ag, err := h.factory.AgentFromRequest(&req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to create agent", err.Error())
		return
	}

	// Build completion request with history
	completionReq := &agent.CompletionRequest{
		Prompt:           req.Prompt,
		SystemPrompt:     req.SystemPrompt,
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		FrequencyPenalty: req.FrequencyPenalty,
		PresencePenalty:  req.PresencePenalty,
		Stop:             req.Stop,
		ChatHistory:      chatHistory,
		ChatID:           &sessionID,
	}

	// Execute completion
	resp, err := ag.CompleteText(r.Context(), completionReq)
	if err != nil {
		log.Error().Err(err).Msg("Completion failed")
		writeError(w, http.StatusInternalServerError, "Completion failed", err.Error())
		return
	}

	// Save messages
	userMsg := &models.ChatMessage{
		SessionID: sessionID,
		Role:      "user",
		Content:   req.Prompt,
	}
	if err := h.chatRepo.AddMessage(r.Context(), userMsg); err != nil {
		log.Error().Err(err).Msg("Failed to save user message")
	}

	assistantMsg := &models.ChatMessage{
		SessionID: sessionID,
		Role:      "assistant",
		Content:   resp.Content,
	}
	if err := h.chatRepo.AddMessage(r.Context(), assistantMsg); err != nil {
		log.Error().Err(err).Msg("Failed to save assistant message")
	}

	// Build response
	apiResp := models.NewCompleteTextResponse(
		resp.ID,
		resp.Model,
		resp.Content,
		resp.FinishReason,
		models.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	)
	apiResp.ChatID = sessionID

	writeJSON(w, http.StatusOK, apiResp)
}

// GetChatHistory handles GET /agent/chat_history/{session_id}
func (h *AgentHandler) GetChatHistory(w http.ResponseWriter, r *http.Request) {
	sessionIDStr := chi.URLParam(r, "session_id")
	sessionID, err := strconv.ParseInt(sessionIDStr, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "Invalid session ID", err.Error())
		return
	}

	limit := 100
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil {
			limit = parsed
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil {
			offset = parsed
		}
	}

	messages, err := h.chatRepo.GetMessages(r.Context(), sessionID, limit, offset)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to get chat history", err.Error())
		return
	}

	total, err := h.chatRepo.GetMessageCount(r.Context(), sessionID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to get message count", err.Error())
		return
	}

	// Convert to response
	var respMessages []models.Message
	for _, msg := range messages {
		respMessages = append(respMessages, msg.ToMessage())
	}

	resp := models.ChatHistoryResponse{
		SessionID: sessionIDStr,
		Messages:  respMessages,
		Total:     total,
	}

	writeJSON(w, http.StatusOK, resp)
}

// InitializeAgent handles POST /agent/initialize_task_and_tick
func (h *AgentHandler) InitializeAgent(w http.ResponseWriter, r *http.Request) {
	var req models.InitializeAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	// Create a new chat session
	session := &models.ChatSession{
		UserID:    1, // TODO: Get from auth context
		AgentType: req.AgentType,
		Model:     req.Model,
		Title:     truncate(req.Prompt, 100),
	}

	if err := h.chatRepo.Create(r.Context(), session); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to create session", err.Error())
		return
	}

	// Get agent
	ag, err := h.factory.GetAgent(req.AgentType, req.Model)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to create agent", err.Error())
		return
	}

	// Initialize agent
	if err := ag.Initialize(r.Context(), req.Prompt); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to initialize agent", err.Error())
		return
	}

	resp := models.InitializeAgentResponse{
		ID:        agent.GenerateID(),
		Status:    "initialized",
		SessionID: strconv.FormatInt(session.ID, 10),
	}

	writeJSON(w, http.StatusOK, resp)
}

// CreateSession handles POST /agent/sessions
func (h *AgentHandler) CreateSession(w http.ResponseWriter, r *http.Request) {
	var req struct {
		AgentType models.AgentType `json:"agent_type"`
		Model     string           `json:"model"`
		Title     string           `json:"title"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", err.Error())
		return
	}

	session := &models.ChatSession{
		UserID:    1, // TODO: Get from auth context
		AgentType: req.AgentType,
		Model:     req.Model,
		Title:     req.Title,
	}

	if err := h.chatRepo.Create(r.Context(), session); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to create session", err.Error())
		return
	}

	writeJSON(w, http.StatusCreated, session)
}

// ListSessions handles GET /agent/sessions
func (h *AgentHandler) ListSessions(w http.ResponseWriter, r *http.Request) {
	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil {
			limit = parsed
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil {
			offset = parsed
		}
	}

	userID := int64(1) // TODO: Get from auth context

	sessions, err := h.chatRepo.GetByUserID(r.Context(), userID, limit, offset)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to list sessions", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, sessions)
}

// DeleteSession handles DELETE /agent/sessions/{session_id}
func (h *AgentHandler) DeleteSession(w http.ResponseWriter, r *http.Request) {
	sessionIDStr := chi.URLParam(r, "session_id")
	sessionID, err := strconv.ParseInt(sessionIDStr, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "Invalid session ID", err.Error())
		return
	}

	if err := h.chatRepo.Delete(r.Context(), sessionID); err != nil {
		writeError(w, http.StatusInternalServerError, "Failed to delete session", err.Error())
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// Helper functions

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, message, detail string) {
	resp := models.ErrorResponse{
		Error: models.ErrorDetail{
			Message: message,
			Type:    http.StatusText(status),
		},
	}
	if detail != "" {
		resp.Error.Code = detail
	}
	writeJSON(w, status, resp)
}

func writeSSEError(w http.ResponseWriter, flusher http.Flusher, message string) {
	errData := map[string]string{"error": message}
	data, _ := json.Marshal(errData)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
