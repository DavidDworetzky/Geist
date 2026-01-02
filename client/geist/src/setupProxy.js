const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Use REACT_APP_BACKEND_HOST env var, or default to localhost for local dev
  // In Docker, set REACT_APP_BACKEND_HOST=host.docker.internal
  const backendHost = process.env.REACT_APP_BACKEND_HOST || 'localhost';
  const backendPort = process.env.REACT_APP_BACKEND_PORT || '5001';
  const target = `http://${backendHost}:${backendPort}`;

  console.log(`Proxying API requests to: ${target}`);

  const proxyConfig = {
    target: target,
    changeOrigin: true,
  };

  // Proxy both /api and /agent paths
  app.use('/api', createProxyMiddleware(proxyConfig));
  app.use('/agent', createProxyMiddleware(proxyConfig));
};
