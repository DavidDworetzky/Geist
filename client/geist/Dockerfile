# Use Node.js 24.22.1 as the base image
FROM node:24.22.1-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY client/geist/package*.json ./

# Install dependencies from the committed lockfile without lifecycle scripts
RUN npm ci --ignore-scripts --audit=false --fund=false

# Copy the rest of the client code
COPY client/geist/ .

# Build the app
RUN npm run build

# The app binds to port 3000 by default
EXPOSE 3000

# Drop root privileges for the runtime process and report service readiness.
USER 1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["wget", "--quiet", "--tries=1", "--spider", "http://127.0.0.1:3000/"]

# Run the app
CMD ["npm", "start"]
