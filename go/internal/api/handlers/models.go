// Package handlers provides HTTP request handlers for the API.
package handlers

import (
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/DavidDworetzky/Geist/internal/agent"
	"github.com/DavidDworetzky/Geist/internal/models"
)

// ModelsHandler handles model-related HTTP requests.
type ModelsHandler struct {
	factory  *agent.Factory
	registry *models.ModelRegistry
}

// NewModelsHandler creates a new ModelsHandler.
func NewModelsHandler(factory *agent.Factory, registry *models.ModelRegistry) *ModelsHandler {
	return &ModelsHandler{
		factory:  factory,
		registry: registry,
	}
}

// ListModels handles GET /api/v1/models
func (h *ModelsHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	// Get query parameters
	provider := r.URL.Query().Get("provider")

	var modelList []models.ModelInfo
	if provider != "" {
		modelList = h.registry.ListByProvider(models.Provider(provider))
	} else {
		modelList = h.registry.List()
	}

	// Filter to only available providers
	availableProviders := make(map[models.Provider]bool)
	for _, p := range h.factory.ListAvailableProviders() {
		availableProviders[p] = true
	}

	var available []models.ModelInfo
	for _, m := range modelList {
		if availableProviders[m.Provider] {
			available = append(available, m)
		}
	}

	resp := struct {
		Object string             `json:"object"`
		Data   []models.ModelInfo `json:"data"`
	}{
		Object: "list",
		Data:   available,
	}

	writeJSON(w, http.StatusOK, resp)
}

// GetModel handles GET /api/v1/models/{model_id}
func (h *ModelsHandler) GetModel(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "model_id")

	model, ok := h.registry.Get(modelID)
	if !ok {
		writeError(w, http.StatusNotFound, "Model not found", modelID)
		return
	}

	writeJSON(w, http.StatusOK, model)
}

// ListProviders handles GET /api/v1/providers
func (h *ModelsHandler) ListProviders(w http.ResponseWriter, r *http.Request) {
	providers := h.factory.ListAvailableProviders()

	resp := struct {
		Providers []models.Provider `json:"providers"`
	}{
		Providers: providers,
	}

	writeJSON(w, http.StatusOK, resp)
}

// LocalModels handles GET /api/v1/models/local
func (h *ModelsHandler) LocalModels(w http.ResponseWriter, r *http.Request) {
	modelList := h.registry.ListByProvider(models.ProviderLocal)

	resp := struct {
		Object string             `json:"object"`
		Data   []models.ModelInfo `json:"data"`
	}{
		Object: "list",
		Data:   modelList,
	}

	writeJSON(w, http.StatusOK, resp)
}
