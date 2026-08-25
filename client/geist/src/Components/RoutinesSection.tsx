import React, { useState, useEffect, useCallback } from 'react';
import SettingsToggle from './SettingsToggle';

export interface Routine {
  routine_id: number;
  name: string;
  prompt: string;
  interval_minutes: number;
  enabled: boolean;
  last_run_at: string | null;
  next_run_at: string | null;
}

const RoutinesSection: React.FC = () => {
  const [routines, setRoutines] = useState<Routine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [intervalMinutes, setIntervalMinutes] = useState(60);
  const [saving, setSaving] = useState(false);

  const fetchRoutines = useCallback(async () => {
    try {
      setError(null);
      const response = await fetch('/api/v1/routines/');
      if (!response.ok) throw new Error(response.statusText);
      setRoutines(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load routines');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchRoutines();
  }, [fetchRoutines]);

  const createRoutine = async () => {
    if (!name.trim() || !prompt.trim()) return;
    try {
      setSaving(true);
      setError(null);
      const response = await fetch('/api/v1/routines/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          prompt: prompt.trim(),
          interval_minutes: intervalMinutes
        })
      });
      if (!response.ok) throw new Error(response.statusText);
      setName('');
      setPrompt('');
      await fetchRoutines();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create routine');
    } finally {
      setSaving(false);
    }
  };

  const toggleRoutine = async (routine: Routine, enabled: boolean) => {
    try {
      const response = await fetch(`/api/v1/routines/${routine.routine_id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      if (!response.ok) throw new Error(response.statusText);
      await fetchRoutines();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update routine');
    }
  };

  const removeRoutine = async (routine: Routine) => {
    if (!window.confirm(`Delete routine "${routine.name}"?`)) return;
    try {
      const response = await fetch(`/api/v1/routines/${routine.routine_id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error(response.statusText);
      await fetchRoutines();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete routine');
    }
  };

  const runNow = async (routine: Routine) => {
    try {
      const response = await fetch(`/api/v1/routines/${routine.routine_id}/run_now`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error(response.statusText);
      await fetchRoutines();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to schedule routine');
    }
  };

  return (
    <section className="settings-section">
      <header className="settings-section-header">
        <h3>Scheduled Routines</h3>
        <p>
          Recurring prompts the agent runs unattended. Tools that require approval are
          denied during routine runs; results appear as chat sessions.
        </p>
      </header>

      {error && <div className="notice notice-error">{error}</div>}

      <div className="settings-subsection">
        <span className="settings-label">New Routine</span>
        <div className="settings-field">
          <label className="settings-label" htmlFor="routine-name">Name</label>
          <input
            id="routine-name"
            className="form-control"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Morning digest"
          />
        </div>
        <div className="settings-field">
          <label className="settings-label" htmlFor="routine-prompt">Prompt</label>
          <textarea
            id="routine-prompt"
            className="form-control"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Summarize my uploaded documents that changed this week."
            rows={3}
          />
        </div>
        <div className="settings-field">
          <label className="settings-label" htmlFor="routine-interval">Interval (minutes)</label>
          <input
            id="routine-interval"
            className="form-control"
            type="number"
            min={5}
            value={intervalMinutes}
            onChange={(e) => setIntervalMinutes(Number(e.target.value) || 60)}
          />
        </div>
        <button
          className="button"
          onClick={createRoutine}
          disabled={saving || !name.trim() || !prompt.trim()}
        >
          {saving ? 'Creating...' : 'Create Routine'}
        </button>
      </div>

      <div className="settings-subsection">
        <span className="settings-label">Your Routines</span>
        {loading ? (
          <div className="empty-state compact">Loading routines...</div>
        ) : routines.length === 0 ? (
          <div className="empty-state compact">No routines yet.</div>
        ) : (
          routines.map((routine) => (
            <div
              key={routine.routine_id}
              data-testid={`routine-${routine.routine_id}`}
              className="settings-field"
              style={{
                border: '1px solid var(--geist-color-border)',
                borderRadius: 'var(--geist-radius-lg)',
                padding: '10px 12px'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <strong>{routine.name}</strong>
                <span className="settings-description">every {routine.interval_minutes} min</span>
              </div>
              <p className="settings-description" style={{ marginTop: 4 }}>{routine.prompt}</p>
              {routine.next_run_at && (
                <p className="settings-description">Next run: {routine.next_run_at}</p>
              )}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
                <SettingsToggle
                  label="Enabled"
                  checked={routine.enabled}
                  onChange={(value) => toggleRoutine(routine, value)}
                />
                <button className="button button-secondary button-small" onClick={() => runNow(routine)}>
                  Run Now
                </button>
                <button className="button button-danger button-small" onClick={() => removeRoutine(routine)}>
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
};

export default RoutinesSection;
