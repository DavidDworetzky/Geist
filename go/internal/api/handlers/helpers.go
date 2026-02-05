// Package handlers provides HTTP request handlers for the API.
package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

// WriteJSON writes a JSON response.
func WriteJSON(w http.ResponseWriter, status int, data interface{}) {
	writeJSON(w, status, data)
}

// DecodeJSON decodes a JSON request body.
func DecodeJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

// ParseInt64Param parses an int64 URL parameter.
func ParseInt64Param(r *http.Request, name string) int64 {
	str := chi.URLParam(r, name)
	if str == "" {
		return 0
	}
	val, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		return 0
	}
	return val
}

// ParseIntQuery parses an int query parameter with a default value.
func ParseIntQuery(r *http.Request, name string, defaultVal int) int {
	str := r.URL.Query().Get(name)
	if str == "" {
		return defaultVal
	}
	val, err := strconv.Atoi(str)
	if err != nil {
		return defaultVal
	}
	return val
}

// ParseBoolQuery parses a bool query parameter with a default value.
func ParseBoolQuery(r *http.Request, name string, defaultVal bool) bool {
	str := r.URL.Query().Get(name)
	if str == "" {
		return defaultVal
	}
	val, err := strconv.ParseBool(str)
	if err != nil {
		return defaultVal
	}
	return val
}
