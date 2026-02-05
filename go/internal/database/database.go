// Package database provides database connection and query functionality.
package database

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog/log"

	"github.com/DavidDworetzky/Geist/internal/config"
)

// DB wraps a pgxpool.Pool with additional functionality.
type DB struct {
	pool *pgxpool.Pool
	cfg  config.DatabaseConfig
}

// New creates a new database connection pool.
func New(ctx context.Context, cfg config.DatabaseConfig) (*DB, error) {
	poolCfg, err := pgxpool.ParseConfig(cfg.URL())
	if err != nil {
		return nil, fmt.Errorf("parsing database config: %w", err)
	}

	poolCfg.MaxConns = int32(cfg.MaxOpenConns)
	poolCfg.MinConns = int32(cfg.MaxIdleConns)
	poolCfg.MaxConnLifetime = cfg.ConnMaxLifetime
	poolCfg.MaxConnIdleTime = cfg.ConnMaxIdleTime

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		return nil, fmt.Errorf("creating connection pool: %w", err)
	}

	// Verify connection
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pinging database: %w", err)
	}

	log.Info().
		Str("host", cfg.Host).
		Int("port", cfg.Port).
		Str("database", cfg.Name).
		Msg("Connected to database")

	return &DB{pool: pool, cfg: cfg}, nil
}

// Close closes the database connection pool.
func (db *DB) Close() {
	if db.pool != nil {
		db.pool.Close()
	}
}

// Pool returns the underlying connection pool.
func (db *DB) Pool() *pgxpool.Pool {
	return db.pool
}

// Ping verifies the database connection.
func (db *DB) Ping(ctx context.Context) error {
	return db.pool.Ping(ctx)
}

// BeginTx starts a new transaction.
func (db *DB) BeginTx(ctx context.Context) (pgx.Tx, error) {
	return db.pool.Begin(ctx)
}

// Query executes a query that returns rows.
func (db *DB) Query(ctx context.Context, sql string, args ...interface{}) (pgx.Rows, error) {
	return db.pool.Query(ctx, sql, args...)
}

// QueryRow executes a query that returns a single row.
func (db *DB) QueryRow(ctx context.Context, sql string, args ...interface{}) pgx.Row {
	return db.pool.QueryRow(ctx, sql, args...)
}

// Exec executes a query that doesn't return rows.
func (db *DB) Exec(ctx context.Context, sql string, args ...interface{}) (int64, error) {
	result, err := db.pool.Exec(ctx, sql, args...)
	if err != nil {
		return 0, err
	}
	return result.RowsAffected(), nil
}

// WithTimeout executes a function with a context timeout.
func (db *DB) WithTimeout(ctx context.Context, timeout time.Duration, fn func(ctx context.Context) error) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return fn(ctx)
}

// Health returns database health information.
func (db *DB) Health(ctx context.Context) (map[string]interface{}, error) {
	stats := db.pool.Stat()

	health := map[string]interface{}{
		"status":          "up",
		"total_conns":     stats.TotalConns(),
		"acquired_conns":  stats.AcquiredConns(),
		"idle_conns":      stats.IdleConns(),
		"max_conns":       stats.MaxConns(),
		"constructing":    stats.ConstructingConns(),
	}

	// Verify connection
	if err := db.Ping(ctx); err != nil {
		health["status"] = "down"
		health["error"] = err.Error()
	}

	return health, nil
}

// Migrate runs database migrations.
func (db *DB) Migrate(ctx context.Context) error {
	migrations := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id BIGSERIAL PRIMARY KEY,
			email VARCHAR(255) UNIQUE NOT NULL,
			password_hash VARCHAR(255),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS chat_sessions (
			id BIGSERIAL PRIMARY KEY,
			user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
			agent_type VARCHAR(50) NOT NULL,
			model VARCHAR(100),
			title VARCHAR(255),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS chat_messages (
			id BIGSERIAL PRIMARY KEY,
			session_id BIGINT REFERENCES chat_sessions(id) ON DELETE CASCADE,
			role VARCHAR(20) NOT NULL,
			content TEXT NOT NULL,
			metadata JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS agent_presets (
			id BIGSERIAL PRIMARY KEY,
			name VARCHAR(100) UNIQUE NOT NULL,
			description TEXT,
			agent_type VARCHAR(50) NOT NULL,
			settings JSONB NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS user_settings (
			id BIGSERIAL PRIMARY KEY,
			user_id BIGINT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
			default_agent_type VARCHAR(50),
			default_model VARCHAR(100),
			default_system_prompt TEXT,
			preferences JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS file_uploads (
			id BIGSERIAL PRIMARY KEY,
			user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
			filename VARCHAR(255) NOT NULL,
			content_type VARCHAR(100),
			size BIGINT NOT NULL,
			storage_path VARCHAR(500) NOT NULL,
			checksum VARCHAR(64),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS workflows (
			id BIGSERIAL PRIMARY KEY,
			user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
			name VARCHAR(100) NOT NULL,
			description TEXT,
			definition JSONB NOT NULL,
			enabled BOOLEAN DEFAULT true,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS workflow_executions (
			id BIGSERIAL PRIMARY KEY,
			workflow_id BIGINT REFERENCES workflows(id) ON DELETE CASCADE,
			status VARCHAR(50) NOT NULL,
			input JSONB,
			output JSONB,
			error TEXT,
			started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			finished_at TIMESTAMP WITH TIME ZONE
		)`,

		`CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id ON file_uploads(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON workflows(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)`,
	}

	for _, migration := range migrations {
		if _, err := db.pool.Exec(ctx, migration); err != nil {
			return fmt.Errorf("running migration: %w", err)
		}
	}

	log.Info().Msg("Database migrations completed")
	return nil
}
