# Use Node.js 20.13.1 as the base image
FROM node:20.13.1-alpine

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

# Run the app
CMD ["npm", "start"]
