import React, { useEffect, useState } from 'react';
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
}

const toolCategory = (tool: ToolDefinition): string => {
  if (tool.name.startsWith('mcp.')) return 'MCP';
  if (tool.semantic_tags?.includes('image')) return 'Image';
  if (tool.semantic_tags?.includes('retrieval')) return 'Retrieval';
  return 'Action';
};

const Tools: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ToolsTab>('catalogue');
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
              <div className="tool-card-grid" data-testid="tool-catalogue">
                {tools.map((tool) => (
                  <article className="tool-card" key={tool.name}>
                    <div className="tool-card-title-row">
                      <div>
                        <span className="tool-category">{toolCategory(tool)}</span>
                        <h3>{tool.name}</h3>
                      </div>
                      <span className={`tool-status ${tool.enabled ? 'tool-status-ready' : ''}`}>
                        {tool.enabled ? 'Available' : 'Disabled'}
                      </span>
                    </div>
                    <p>{tool.description}</p>
                    <div className="tool-meta-row">
                      <span>{tool.name.startsWith('mcp.') ? 'MCP' : 'Built in'}</span>
                      <span>{tool.side_effect.replace('_', ' ')}</span>
                      {tool.requires_approval && <span>Approval required</span>}
                      {!tool.enabled_by_default && <span>Opt-in</span>}
                    </div>
                    {tool.source_adapter && (
                      <p className="tool-source">Source: {tool.source_adapter}</p>
                    )}
                    <details>
                      <summary>Input schema</summary>
                      <pre>{JSON.stringify(tool.input_schema, null, 2)}</pre>
                    </details>
                  </article>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'mcp' && (
          <div className="tools-mcp-panel" role="tabpanel">
            <McpServersSection />
          </div>
        )}
      </div>
    </section>
  );
};

export default Tools;
