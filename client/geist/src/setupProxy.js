const { createProxyMiddleware } = require('http-proxy-middleware');
const fs = require('fs');

function operatorAuthorization() {
  let token = process.env.GEIST_OPERATOR_TOKEN;
  const tokenFile = process.env.GEIST_OPERATOR_TOKEN_FILE;
  if (!token && tokenFile) {
    try {
      token = fs.readFileSync(tokenFile, 'utf8').trim();
    } catch (error) {
      throw new Error('Could not read the Geist operator token file');
    }
  }
  return token ? `GeistOperator ${token}` : null;
}

module.exports = function(app) {
  // Use REACT_APP_BACKEND_HOST env var, or default to localhost for local dev
  // In Docker, set REACT_APP_BACKEND_HOST=host.docker.internal
  const backendHost = process.env.REACT_APP_BACKEND_HOST || 'localhost';
  const backendPort = process.env.REACT_APP_BACKEND_PORT || '5001';
  const target = `http://${backendHost}:${backendPort}`;
  const authorization = operatorAuthorization();

  console.log(`Proxying API requests to: ${target}`);

  const proxyConfig = {
    target: target,
    changeOrigin: true,
    onProxyReq: (proxyRequest) => {
      if (authorization) {
        proxyRequest.setHeader('Authorization', authorization);
      }
    },
  };

  // Proxy both /api and /agent paths
  app.use('/api', createProxyMiddleware(proxyConfig));
  app.use('/agent', createProxyMiddleware(proxyConfig));
};
