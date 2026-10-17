export type McpTransport = 'stdio' | 'http';

export interface McpCatalogueEntry {
  id: string;
  name: string;
  publisher: string;
  description: string;
  category: 'Developer tools' | 'Browser automation' | 'Email';
  sourceUrl: string;
  transport: McpTransport;
  command?: string;
  args?: string[];
  url?: string;
  accountScope?: string;
  authentication?: string;
  requirements: string[];
  configurationNote?: string;
  oauthProvider?: string;
  oauthLabel?: string;
}

export const mcpCatalogue: McpCatalogueEntry[] = [
  {
    id: 'context7',
    name: 'Context7',
    publisher: 'Upstash',
    description: 'Look up current, version-specific documentation for software libraries.',
    category: 'Developer tools',
    sourceUrl: 'https://github.com/upstash/context7',
    transport: 'http',
    url: 'https://mcp.context7.com/mcp',
    requirements: [],
  },
  {
    id: 'github',
    oauthProvider: 'github',
    oauthLabel: 'GitHub',
    name: 'GitHub',
    publisher: 'GitHub',
    description: 'Read repositories, issues, pull requests, and other GitHub context.',
    category: 'Developer tools',
    sourceUrl: 'https://github.com/github/github-mcp-server',
    transport: 'http',
    url: 'https://api.githubcopilot.com/mcp/',
    authentication: 'GitHub personal access token or OAuth token',
    requirements: ['Connect with GitHub using your operator’s OAuth application, or add a personal access token as an Authorization bearer header.'],
  },
  {
    id: 'playwright',
    name: 'Playwright',
    publisher: 'Microsoft',
    description: 'Inspect and automate web pages through structured accessibility snapshots.',
    category: 'Browser automation',
    sourceUrl: 'https://github.com/microsoft/playwright-mcp',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@playwright/mcp@latest'],
    requirements: ['Connecting may cause npx to download and run the publisher package.'],
  },
  {
    id: 'gmail',
    oauthProvider: 'google',
    name: 'Gmail',
    publisher: 'Google',
    description: 'Search, read, label, and draft mail for a personal Google account.',
    category: 'Email',
    sourceUrl: 'https://developers.google.com/workspace/gmail/api/reference/mcp',
    transport: 'http',
    url: 'https://gmailmcp.googleapis.com/mcp/v1',
    accountScope: 'Personal Google Account',
    authentication: 'Google OAuth 2.0 delegated authorization',
    requirements: [
      'Google Workspace Developer Preview access and Gmail MCP configuration.',
      'A Google OAuth application configured by your Geist operator; sign in with Google during setup.',
    ],
    configurationNote: 'Connect with Google to grant mail reading and draft permissions. Geist refreshes authorization automatically. The server stays disabled until you enable it.',
  },
  {
    id: 'google-workspace-mail',
    oauthProvider: 'google',
    name: 'Google Workspace Mail',
    publisher: 'Google',
    description: 'Use Gmail tools with an administrator-managed Google Workspace account.',
    category: 'Email',
    sourceUrl: 'https://developers.google.com/workspace/gmail/api/reference/mcp',
    transport: 'http',
    url: 'https://gmailmcp.googleapis.com/mcp/v1',
    accountScope: 'Managed Google Workspace account',
    authentication: 'Google OAuth 2.0 under Workspace administrator policy',
    requirements: [
      'Google Workspace Developer Preview access and Gmail MCP configuration.',
      'Workspace administrator approval when organizational policy requires it.',
      'A Google OAuth application configured by your Geist operator; sign in with Google during setup.',
    ],
    configurationNote: 'Connect with Google, review the requested permissions, then test the server before enabling it.',
  },
  {
    id: 'outlook-mail',
    name: 'Outlook / Microsoft 365',
    publisher: 'Microsoft',
    description: 'Connect Microsoft 365 mail through an operator-provided MCP endpoint.',
    category: 'Email',
    sourceUrl: 'https://github.com/microsoft/mcp',
    transport: 'http',
    accountScope: 'Personal Outlook or managed Microsoft 365 account',
    authentication: 'Microsoft identity platform OAuth 2.0 delegated authorization',
    requirements: [
      'A tenant-specific Microsoft 365 Mail MCP endpoint or separately vetted Outlook MCP server.',
      'Tenant administrator consent when organizational policy requires it.',
      'An OAuth provider registration configured by your Geist operator for this exact endpoint, or a manual Authorization bearer header.',
    ],
    configurationNote: 'Microsoft does not publish a universal Outlook MCP endpoint here. Enter the endpoint supplied by your operator and keep the server disabled until it has been reviewed.',
  },
  {
    id: 'proton-mail',
    name: 'Proton Mail',
    publisher: 'Proton',
    description: 'Connect a paid Proton account through Proton Mail Bridge and a vetted local MCP server.',
    category: 'Email',
    sourceUrl: 'https://proton.me/support/imap-smtp-and-pop3-setup',
    transport: 'stdio',
    accountScope: 'Paid Proton account using Proton Mail Bridge',
    authentication: 'Bridge-issued local IMAP/SMTP credentials',
    requirements: [
      'Proton Mail Bridge installed, running, and signed in.',
      'A separately installed, operator-vetted IMAP/SMTP MCP server.',
      'Bridge credentials configured in the local server environment.',
    ],
    configurationNote: 'Geist does not install an IMAP/SMTP MCP server. Enter the command for the server you have vetted; the configuration is saved disabled.',
  },
];
