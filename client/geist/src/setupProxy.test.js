jest.mock('http-proxy-middleware', () => ({
  createProxyMiddleware: jest.fn((config) => config),
}));

const { createProxyMiddleware } = require('http-proxy-middleware');
const fs = require('fs');
const setupProxy = require('./setupProxy');

describe('development API proxy operator authentication', () => {
  const originalToken = process.env.GEIST_OPERATOR_TOKEN;
  const originalTokenFile = process.env.GEIST_OPERATOR_TOKEN_FILE;

  afterEach(() => {
    jest.clearAllMocks();
    if (originalToken === undefined) {
      delete process.env.GEIST_OPERATOR_TOKEN;
    } else {
      process.env.GEIST_OPERATOR_TOKEN = originalToken;
    }
    if (originalTokenFile === undefined) {
      delete process.env.GEIST_OPERATOR_TOKEN_FILE;
    } else {
      process.env.GEIST_OPERATOR_TOKEN_FILE = originalTokenFile;
    }
  });

  test('injects the operator token from the server process environment', () => {
    process.env.GEIST_OPERATOR_TOKEN = 't'.repeat(32);
    delete process.env.GEIST_OPERATOR_TOKEN_FILE;
    const app = { use: jest.fn() };

    setupProxy(app);
    const proxyConfig = createProxyMiddleware.mock.calls[0][0];
    const proxyRequest = { setHeader: jest.fn() };
    proxyConfig.onProxyReq(proxyRequest);
    const websocketRequest = { setHeader: jest.fn() };
    proxyConfig.onProxyReqWs(websocketRequest);

    expect(proxyRequest.setHeader).toHaveBeenCalledWith(
      'Authorization',
      `GeistOperator ${'t'.repeat(32)}`,
    );
    expect(websocketRequest.setHeader).toHaveBeenCalledWith(
      'Authorization',
      `GeistOperator ${'t'.repeat(32)}`,
    );
    expect(proxyConfig.ws).toBe(true);
    expect(app.use).toHaveBeenCalledTimes(3);
    expect(app.use.mock.calls.map(([path]) => path)).toEqual([
      '/api',
      '/agent',
      '/adapter',
    ]);
    expect(createProxyMiddleware.mock.calls[1][0].ws).toBeUndefined();
    expect(createProxyMiddleware.mock.calls[2][0].ws).toBeUndefined();
  });

  test('leaves authorization untouched when standalone proxy auth is not configured', () => {
    delete process.env.GEIST_OPERATOR_TOKEN;
    delete process.env.GEIST_OPERATOR_TOKEN_FILE;
    const app = { use: jest.fn() };

    setupProxy(app);
    const proxyConfig = createProxyMiddleware.mock.calls[0][0];
    const proxyRequest = { setHeader: jest.fn() };
    proxyConfig.onProxyReq(proxyRequest);

    expect(proxyRequest.setHeader).not.toHaveBeenCalled();
  });

  test('injects the generated local token from the server-only file', () => {
    delete process.env.GEIST_OPERATOR_TOKEN;
    process.env.GEIST_OPERATOR_TOKEN_FILE = '/run/geist-operator/token';
    const readToken = jest.spyOn(fs, 'readFileSync').mockReturnValue(`${'f'.repeat(32)}\n`);
    const app = { use: jest.fn() };

    setupProxy(app);
    const proxyConfig = createProxyMiddleware.mock.calls[0][0];
    const proxyRequest = { setHeader: jest.fn() };
    proxyConfig.onProxyReq(proxyRequest);

    expect(proxyRequest.setHeader).toHaveBeenCalledWith(
      'Authorization',
      `GeistOperator ${'f'.repeat(32)}`,
    );
    readToken.mockRestore();
  });
});
