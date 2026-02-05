// Package handlers provides HTTP request handlers for the API.
package handlers

import (
	"net/http"
	"runtime"
	"time"

	"github.com/DavidDworetzky/Geist/internal/database"
)

// HealthHandler handles health check requests.
type HealthHandler struct {
	db        *database.DB
	startTime time.Time
	version   string
}

// NewHealthHandler creates a new HealthHandler.
func NewHealthHandler(db *database.DB, version string) *HealthHandler {
	return &HealthHandler{
		db:        db,
		startTime: time.Now(),
		version:   version,
	}
}

// Health handles GET /health
func (h *HealthHandler) Health(w http.ResponseWriter, r *http.Request) {
	resp := map[string]interface{}{
		"status":  "ok",
		"version": h.version,
	}
	writeJSON(w, http.StatusOK, resp)
}

// Ready handles GET /ready
func (h *HealthHandler) Ready(w http.ResponseWriter, r *http.Request) {
	// Check database connection
	if err := h.db.Ping(r.Context()); err != nil {
		resp := map[string]interface{}{
			"status": "not_ready",
			"error":  err.Error(),
		}
		writeJSON(w, http.StatusServiceUnavailable, resp)
		return
	}

	resp := map[string]interface{}{
		"status": "ready",
	}
	writeJSON(w, http.StatusOK, resp)
}

// Live handles GET /live
func (h *HealthHandler) Live(w http.ResponseWriter, r *http.Request) {
	resp := map[string]interface{}{
		"status": "alive",
	}
	writeJSON(w, http.StatusOK, resp)
}

// Status handles GET /status
func (h *HealthHandler) Status(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	dbHealth, _ := h.db.Health(r.Context())

	resp := map[string]interface{}{
		"status":        "ok",
		"version":       h.version,
		"uptime":        time.Since(h.startTime).String(),
		"uptime_seconds": int64(time.Since(h.startTime).Seconds()),
		"go_version":    runtime.Version(),
		"goroutines":    runtime.NumGoroutine(),
		"memory": map[string]interface{}{
			"alloc_mb":       m.Alloc / 1024 / 1024,
			"total_alloc_mb": m.TotalAlloc / 1024 / 1024,
			"sys_mb":         m.Sys / 1024 / 1024,
			"num_gc":         m.NumGC,
		},
		"database": dbHealth,
	}

	writeJSON(w, http.StatusOK, resp)
}
