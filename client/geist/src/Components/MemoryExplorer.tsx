import { FormEvent, useCallback, useEffect, useState } from 'react';
import StagePanelIcon from './StagePanelIcon';


interface ProfileFact {
  memory_id: number;
  content: string;
}

interface SearchResult {
  memory_id: number;
  content: string;
  record_type: string;
  score: number;
}

interface MemoryExplorerProps {
  scope: 'user' | 'folder';
  folderId: number | null;
}


export default function MemoryExplorer({ scope, folderId }: MemoryExplorerProps) {
  const [facts, setFacts] = useState<ProfileFact[]>([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);

  const loadProfile = useCallback(async () => {
    if (scope !== 'user') {
      setFacts([]);
      return;
    }
    const response = await fetch('/api/v1/memory/profile');
    if (!response.ok) throw new Error('Could not load profile memory');
    const profile = await response.json();
    setFacts(profile.facts ?? []);
  }, [scope]);

  useEffect(() => {
    void loadProfile().catch(loadError => {
      setError(loadError instanceof Error ? loadError.message : 'Could not load memory');
    });
    setResults([]);
    setQuery('');
  }, [folderId, loadProfile, scope]);

  const runSearch = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          scope,
          folder_id: scope === 'folder' ? folderId : null,
          limit: 6,
        }),
      });
      if (!response.ok) throw new Error('Memory search failed');
      const body = await response.json();
      setResults(body.results ?? []);
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : 'Memory search failed');
    } finally {
      setSearching(false);
    }
  };

  const saveFact = async (memoryId: number) => {
    const response = await fetch(`/api/v1/memory/records/${memoryId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: draft }),
    });
    if (!response.ok) {
      setError('Could not update this memory');
      return;
    }
    setEditingId(null);
    setDraft('');
    await loadProfile();
  };

  const deleteFact = async (memoryId: number) => {
    const response = await fetch(`/api/v1/memory/records/${memoryId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      setError('Could not forget this memory');
      return;
    }
    await loadProfile();
  };

  return (
    <details className="memory-explorer">
      <summary>
        <span>
          <strong>{scope === 'user' ? 'Profile memory' : 'Folder memory'}</strong>
          <small>Review and search saved context</small>
        </span>
        <StagePanelIcon name="search" />
      </summary>
      <div className="memory-explorer-body">
        <form className="memory-search-form" onSubmit={runSearch}>
          <input
            aria-label="Search memory"
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder={scope === 'user' ? 'Search global memory' : 'Search this folder'}
          />
          <button type="submit" aria-label="Run memory search" disabled={searching}>
            <StagePanelIcon name="search" />
          </button>
        </form>

        {scope === 'user' && facts.length > 0 && (
          <div className="memory-fact-list" aria-label="Saved profile facts">
            {facts.map(fact => (
              <div className="memory-fact" key={fact.memory_id}>
                {editingId === fact.memory_id ? (
                  <>
                    <input
                      aria-label="Edit memory"
                      value={draft}
                      onChange={event => setDraft(event.target.value)}
                    />
                    <button
                      type="button"
                      aria-label="Save memory"
                      onClick={() => void saveFact(fact.memory_id)}
                    >
                      <StagePanelIcon name="check" />
                    </button>
                  </>
                ) : (
                  <>
                    <p>{fact.content}</p>
                    <button
                      type="button"
                      aria-label="Edit saved memory"
                      onClick={() => {
                        setEditingId(fact.memory_id);
                        setDraft(fact.content);
                      }}
                    >
                      <StagePanelIcon name="edit" />
                    </button>
                    <button
                      type="button"
                      aria-label="Forget saved memory"
                      onClick={() => void deleteFact(fact.memory_id)}
                    >
                      <StagePanelIcon name="close" />
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {results.length > 0 && (
          <div className="memory-search-results" aria-label="Memory search results">
            {results.map(result => (
              <article key={result.memory_id}>
                <span>{result.record_type.replace(/_/g, ' ')}</span>
                <p>{result.content}</p>
              </article>
            ))}
          </div>
        )}

        {!error && scope === 'user' && facts.length === 0 && results.length === 0 && (
          <p className="memory-explorer-empty">No global facts saved yet.</p>
        )}
        {error && <p className="memory-explorer-error" role="alert">{error}</p>}
      </div>
    </details>
  );
}
