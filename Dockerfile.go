# Build stage for Go server
FROM golang:1.22-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app

# Copy go mod files first for better caching
COPY go/go.mod go/go.sum ./go/

WORKDIR /app/go
RUN go mod download

# Copy source code
WORKDIR /app
COPY go/ ./go/

# Build the server
WORKDIR /app/go
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.version=$(git describe --tags --always 2>/dev/null || echo 'dev')" \
    -o /geist-server \
    ./cmd/server

# Final stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN addgroup -g 1000 geist && \
    adduser -u 1000 -G geist -s /bin/sh -D geist

# Copy binary from builder
COPY --from=builder /geist-server /usr/local/bin/geist-server

# Set ownership
RUN chown geist:geist /usr/local/bin/geist-server

# Switch to non-root user
USER geist

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Set environment defaults
ENV SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    LOG_LEVEL=info \
    LOG_FORMAT=json

# Run the server
ENTRYPOINT ["/usr/local/bin/geist-server"]
