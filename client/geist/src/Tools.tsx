import React, { useEffect, useRef, useState } from 'react';
import McpServersSection from './Components/McpServersSection';
import './Tools.css';

type ToolsTab = 'catalogue' | 'mcp';

interface ToolDefinition {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  enabled: boolean;
  enabled_by_default: boolean;
  requires_approval: boolean;
  side_effect: string;
  source_adapter?: string | null;
  semantic_tags?: string[];
  configuration?: ToolConfiguration | null;
}

interface ToolConfiguration {
  kind: 'environment';
  provider: string;
  api_key_configured: boolean;
  base_url: string;
  model: string;
  environment_variables: {
    api_key: string;
    base_url: string;
    model: string;
  };
}

const toolCategory = (tool: ToolDefinition): string => {
  if (tool.name.startsWith('mcp.')) return 'MCP';
  if (tool.semantic_tags?.includes('image_generation')) return 'Image';
  if (tool.semantic_tags?.includes('public_retrieval')) return 'Public retrieval';
  if (tool.semantic_tags?.includes('local_retrieval')) return 'Local retrieval';
  return 'Action';
};

const toolStatus = (tool: ToolDefinition): { label: string; className: string; note: string } => {
  if (tool.enabled) {
    return { label: 'Available', className: 'tool-status-ready', note: 'Callable now.' };
  }
  if (tool.enabled_by_default) {
    return {
      label: 'Unavailable',
      className: 'tool-status-unavailable',
      note: 'Enabled by default, but a provider or local prerequisite is not configured.',
    };
  }
  return {
    label: 'Opt-in',
    className: 'tool-status-opt-in',
    note: 'Disabled by default and must be explicitly enabled in Geist configuration.',
  };
};

const Tools: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ToolsTab>('catalogue');
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [configurationTool, setConfigurationTool] = useState<ToolDefinition | null>(null);
  const closeConfigurationButton = useRef<HTMLButtonElement>(null);

  const loadTools = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('/agent/tools');
      if (!response.ok) {
        throw new Error(`Tool catalogue request failed with status ${response.status}`);
      }
      const payload = await response.json();
      setTools(Array.isArray(payload?.tools) ? payload.tools : []);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'The tool catalogue could not be loaded.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadTools();
  }, []);

  useEffect(() => {
    if (!configurationTool) return undefined;
    closeConfigurationButton.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setConfigurationTool(null);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [configurationTool]);

  return (
    <section className="tools-page page-surface">
      <header className="tools-page-header">
        <div>
          <p className="section-eyebrow">Capabilities</p>
          <h1>Tools</h1>
          <p>Inspect the tools available to Geist and configure external MCP servers.</p>
        </div>
        {activeTab === 'catalogue' && (
          <button className="button button-small" type="button" onClick={() => void loadTools()}>
            Refresh
          </button>
        )}
      </header>

      <div className="settings-tabs tools-tabs" role="tablist" aria-label="Tool sections">
        <button
          className={`settings-tab ${activeTab === 'catalogue' ? 'active' : ''}`}
          type="button"
          role="tab"
          aria-selected={activeTab === 'catalogue'}
          onClick={() => setActiveTab('catalogue')}
        >
          Tool catalogue
        </button>
        <button
          className={`settings-tab ${activeTab === 'mcp' ? 'active' : ''}`}
          type="button"
          role="tab"
          aria-selected={activeTab === 'mcp'}
          onClick={() => setActiveTab('mcp')}
        >
          MCP servers
        </button>
      </div>

      <div className="tools-scroll-region">
        {activeTab === 'catalogue' && (
          <div role="tabpanel">
            {error && <div className="notice notice-error">{error}</div>}
            {loading ? (
              <div className="empty-state">Loading tool catalogue...</div>
            ) : tools.length === 0 ? (
              <div className="empty-state">No tools are registered.</div>
            ) : (
              <>
                <div className="tool-status-legend" aria-label="Tool status guide">
                  <span><strong>Available</strong> callable now</span>
                  <span><strong>Opt-in</strong> disabled by default</span>
                  <span><strong>Unavailable</strong> missing a provider or prerequisite</span>
                </div>
                <div className="tool-card-grid" data-testid="tool-catalogue">
                  {tools.map((tool) => {
                    const status = toolStatus(tool);
                    return (
                      <article className="tool-card" key={tool.name}>
                        <div className="tool-card-title-row">
                          <div>
                            <span className="tool-category">{toolCategory(tool)}</span>
                            <h3>{tool.name}</h3>
                          </div>
                          <span className={`tool-status ${status.className}`} title={status.note}>
                            {status.label}
                          </span>
                        </div>
                        <p>{tool.description}</p>
                        {!tool.enabled && <p className="tool-status-note">{status.note}</p>}
                        <div className="tool-meta-row">
                          <span>{tool.name.startsWith('mcp.') ? 'MCP' : 'Built in'}</span>
                          <span>{tool.side_effect.replace('_', ' ')}</span>
                          {tool.requires_approval && <span>Approval required</span>}
                        </div>
                        {tool.source_adapter && (
                          <p className="tool-source">Source: {tool.source_adapter}</p>
                        )}
                        {tool.configuration && (
                          <button
                            className="button button-secondary button-small tool-configure-button"
                            type="button"
                            onClick={() => setConfigurationTool(tool)}
                          >
                            Configure
                          </button>
                        )}
                        <details>
                          <summary>Input schema</summary>
                          <pre>{JSON.stringify(tool.input_schema, null, 2)}</pre>
                        </details>
                      </article>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'mcp' && (
          <div className="tools-mcp-panel" role="tabpanel">
            <McpServersSection />
          </div>
        )}
      </div>

      {configurationTool?.configuration && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setConfigurationTool(null);
          }}
        >
          <div
            className="modal-panel tool-configuration-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="tool-configuration-title"
          >
            <div className="tool-configuration-heading">
              <div>
                <p className="section-eyebrow">Built-in tool</p>
                <h3 id="tool-configuration-title">Configure {configurationTool.name}</h3>
              </div>
              <button
                ref={closeConfigurationButton}
                className="button button-secondary button-small"
                type="button"
                onClick={() => setConfigurationTool(null)}
              >
                Close
              </button>
            </div>

            <p>
              Image generation is configured through Geist&apos;s process environment.
              Secret values are never returned to this page.
            </p>

            <div className="tool-configuration-fields">
              <label>
                <span>Provider</span>
                <input readOnly value={configurationTool.configuration.provider} />
              </label>
              <label>
                <span>API key</span>
                <input
                  readOnly
                  value={
                    configurationTool.configuration.api_key_configured
                      ? 'Configured'
                      : 'Not configured'
                  }
                />
                <small>{configurationTool.configuration.environment_variables.api_key}</small>
              </label>
              <label>
                <span>Base URL</span>
                <input readOnly value={configurationTool.configuration.base_url} />
                <small>
                  Override with {configurationTool.configuration.environment_variables.base_url}
                </small>
              </label>
              <label>
                <span>Model</span>
                <input readOnly value={configurationTool.configuration.model} />
                <small>
                  Override with {configurationTool.configuration.environment_variables.model}
                </small>
              </label>
            </div>

            {!configurationTool.configuration.api_key_configured && (
              <div className="notice tool-configuration-notice">
                Set {configurationTool.configuration.environment_variables.api_key} before
                starting Geist, then restart the app and refresh this catalogue.
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
};

export default Tools;
