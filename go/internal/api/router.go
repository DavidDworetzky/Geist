// Package api provides the HTTP API server implementation.
package api

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"

	"github.com/DavidDworetzky/Geist/internal/agent"
	"github.com/DavidDworetzky/Geist/internal/api/handlers"
	"github.com/DavidDworetzky/Geist/internal/api/middleware"
	"github.com/DavidDworetzky/Geist/internal/config"
	"github.com/DavidDworetzky/Geist/internal/database"
	"github.com/DavidDworetzky/Geist/internal/database/repository"
	"github.com/DavidDworetzky/Geist/internal/models"
)

// Server represents the HTTP API server.
type Server struct {
	router   *chi.Mux
	cfg      *config.Config
	db       *database.DB
	factory  *agent.Factory
	registry *models.ModelRegistry
}

// NewServer creates a new API server.
func NewServer(cfg *config.Config, db *database.DB, factory *agent.Factory) *Server {
	s := &Server{
		router:   chi.NewRouter(),
		cfg:      cfg,
		db:       db,
		factory:  factory,
		registry: models.DefaultRegistry,
	}

	s.setupMiddleware()
	s.setupRoutes()

	return s
}

func (s *Server) setupMiddleware() {
	// Basic middleware
	s.router.Use(chimiddleware.RealIP)
	s.router.Use(middleware.RequestID)
	s.router.Use(middleware.Logger)
	s.router.Use(middleware.Recoverer)

	// CORS
	corsConfig := middleware.DefaultCORSConfig()
	s.router.Use(middleware.CORS(corsConfig))

	// Request limits
	s.router.Use(middleware.MaxBodySize(s.cfg.Server.MaxRequestSize))
}

func (s *Server) setupRoutes() {
	// Initialize repositories
	chatRepo := repository.NewChatSessionRepository(s.db.Pool())
	presetRepo := repository.NewAgentPresetRepository(s.db.Pool())
	settingsRepo := repository.NewUserSettingsRepository(s.db.Pool())

	// Initialize handlers
	agentHandler := handlers.NewAgentHandler(s.factory, chatRepo)
	modelsHandler := handlers.NewModelsHandler(s.factory, s.registry)
	settingsHandler := handlers.NewUserSettingsHandler(settingsRepo)
	healthHandler := handlers.NewHealthHandler(s.db, "1.0.0")

	// Health endpoints (no authentication)
	s.router.Get("/health", healthHandler.Health)
	s.router.Get("/ready", healthHandler.Ready)
	s.router.Get("/live", healthHandler.Live)
	s.router.Get("/status", healthHandler.Status)

	// Agent routes
	s.router.Route("/agent", func(r chi.Router) {
		r.Post("/complete_text", agentHandler.CompleteText)
		r.Post("/stream_complete_text", agentHandler.StreamCompleteText)
		r.Post("/complete_text/{session_id}", agentHandler.CompleteTextWithSession)
		r.Get("/chat_history/{session_id}", agentHandler.GetChatHistory)
		r.Post("/initialize_task_and_tick", agentHandler.InitializeAgent)

		// Session management
		r.Post("/sessions", agentHandler.CreateSession)
		r.Get("/sessions", agentHandler.ListSessions)
		r.Delete("/sessions/{session_id}", agentHandler.DeleteSession)
	})

	// API v1 routes
	s.router.Route("/api/v1", func(r chi.Router) {
		// Models
		r.Route("/models", func(r chi.Router) {
			r.Get("/", modelsHandler.ListModels)
			r.Get("/local", modelsHandler.LocalModels)
			r.Get("/{model_id}", modelsHandler.GetModel)
		})

		// Providers
		r.Get("/providers", modelsHandler.ListProviders)

		// User settings
		r.Route("/user-settings", func(r chi.Router) {
			r.Get("/", settingsHandler.GetSettings)
			r.Put("/", settingsHandler.UpdateSettings)
			r.Patch("/", settingsHandler.PatchSettings)
			r.Delete("/", settingsHandler.ResetSettings)
		})

		// Agent presets
		r.Route("/presets", func(r chi.Router) {
			r.Get("/", s.listPresets(presetRepo))
			r.Post("/", s.createPreset(presetRepo))
			r.Get("/{preset_id}", s.getPreset(presetRepo))
			r.Put("/{preset_id}", s.updatePreset(presetRepo))
			r.Delete("/{preset_id}", s.deletePreset(presetRepo))
		})
	})
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	return s.router
}

// Preset handlers (inline for simplicity)

func (s *Server) listPresets(repo *repository.AgentPresetRepository) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		presets, err := repo.List(r.Context())
		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, models.NewErrorResponse("Failed to list presets", "internal_error"))
			return
		}
		handlers.WriteJSON(w, http.StatusOK, presets)
	}
}

func (s *Server) createPreset(repo *repository.AgentPresetRepository) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var preset models.AgentPreset
		if err := handlers.DecodeJSON(r, &preset); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, models.NewErrorResponse("Invalid request body", "invalid_request"))
			return
		}

		if err := repo.Create(r.Context(), &preset); err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, models.NewErrorResponse("Failed to create preset", "internal_error"))
			return
		}

		handlers.WriteJSON(w, http.StatusCreated, preset)
	}
}

func (s *Server) getPreset(repo *repository.AgentPresetRepository) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		presetID := handlers.ParseInt64Param(r, "preset_id")
		if presetID == 0 {
			handlers.WriteJSON(w, http.StatusBadRequest, models.NewErrorResponse("Invalid preset ID", "invalid_request"))
			return
		}

		preset, err := repo.GetByID(r.Context(), presetID)
		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, models.NewErrorResponse("Failed to get preset", "internal_error"))
			return
		}
		if preset == nil {
			handlers.WriteJSON(w, http.StatusNotFound, models.NewErrorResponse("Preset not found", "not_found"))
			return
		}

		handlers.WriteJSON(w, http.StatusOK, preset)
	}
}

func (s *Server) updatePreset(repo *repository.AgentPresetRepository) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		presetID := handlers.ParseInt64Param(r, "preset_id")
		if presetID == 0 {
			handlers.WriteJSON(w, http.StatusBadRequest, models.NewErrorResponse("Invalid preset ID", "invalid_request"))
			return
		}

		var preset models.AgentPreset
		if err := handlers.DecodeJSON(r, &preset); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, models.NewErrorResponse("Invalid request body", "invalid_request"))
			return
		}

		preset.ID = presetID
		if err := repo.Update(r.Context(), &preset); err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, models.NewErrorResponse("Failed to update preset", "internal_error"))
			return
		}

		handlers.WriteJSON(w, http.StatusOK, preset)
	}
}

func (s *Server) deletePreset(repo *repository.AgentPresetRepository) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		presetID := handlers.ParseInt64Param(r, "preset_id")
		if presetID == 0 {
			handlers.WriteJSON(w, http.StatusBadRequest, models.NewErrorResponse("Invalid preset ID", "invalid_request"))
			return
		}

		if err := repo.Delete(r.Context(), presetID); err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, models.NewErrorResponse("Failed to delete preset", "internal_error"))
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}
}
