# Use Node.js 20.13.1 as the base image
FROM node:20.13.1-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY client/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the client code
COPY client/ .

# Build the app
RUN npm run build

# Install a simple http server for serving static content
RUN npm install -g serve

# The app binds to port 3000 by default
EXPOSE 3000

# Run the app
CMD ["serve", "-s", "build", "-l", "3000"]