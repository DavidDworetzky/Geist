import React, { useEffect, useState } from 'react';
import './Schedules.css';

interface PromptSchedule {
  prompt_schedule_id: number;
  name: string;
  prompt: string;
  cron_expression: string;
  timezone: string;
  enabled: boolean;
  inference_config: {
    agent_type?: 'local' | 'online';
  };
  next_run_at: string | null;
  last_enqueued_at: string | null;
}

interface ScheduleForm {
  name: string;
  prompt: string;
  cron_expression: string;
  timezone: string;
  agent_type: '' | 'local' | 'online';
}

const EMPTY_FORM: ScheduleForm = {
  name: '',
  prompt: '',
  cron_expression: '0 9 * * 1-5',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
  agent_type: '',
};

function formatUtc(value: string | null): string {
  if (!value) return 'Not scheduled';
  const normalized = /(?:Z|[+-]\d\d:\d\d)$/.test(value) ? value : `${value}Z`;
  return new Date(normalized).toLocaleString();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong';
}

const Schedules: React.FC = () => {
  const [schedules, setSchedules] = useState<PromptSchedule[]>([]);
  const [form, setForm] = useState<ScheduleForm>(EMPTY_FORM);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const fetchSchedules = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/prompt-schedules/');
      if (!response.ok) throw new Error('Unable to load schedules');
      setSchedules(await response.json());
      setError(null);
    } catch (fetchError) {
      setError(errorMessage(fetchError));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchSchedules();
  }, []);

  const resetForm = () => {
    setEditingId(null);
    setForm(EMPTY_FORM);
  };

  const submitSchedule = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch(
        editingId === null
          ? '/api/v1/prompt-schedules/'
          : `/api/v1/prompt-schedules/${editingId}`,
        {
          method: editingId === null ? 'POST' : 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: form.name,
            prompt: form.prompt,
            cron_expression: form.cron_expression,
            timezone: form.timezone,
            inference_config: form.agent_type ? { agent_type: form.agent_type } : {},
          }),
        },
      );
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail?.[0]?.msg || detail?.detail || 'Unable to save schedule');
      }
      setNotice(editingId === null ? 'Schedule created.' : 'Schedule updated.');
      resetForm();
      await fetchSchedules();
    } catch (saveError) {
      setError(errorMessage(saveError));
    } finally {
      setSaving(false);
    }
  };

  const editSchedule = (schedule: PromptSchedule) => {
    setEditingId(schedule.prompt_schedule_id);
    setForm({
      name: schedule.name,
      prompt: schedule.prompt,
      cron_expression: schedule.cron_expression,
      timezone: schedule.timezone,
      agent_type: schedule.inference_config.agent_type || '',
    });
    setNotice(null);
  };

  const mutateSchedule = async (
    schedule: PromptSchedule,
    action: 'toggle' | 'run' | 'delete',
  ) => {
    setError(null);
    setNotice(null);
    const url = action === 'run'
      ? `/api/v1/prompt-schedules/${schedule.prompt_schedule_id}/run`
      : `/api/v1/prompt-schedules/${schedule.prompt_schedule_id}`;
    const options: RequestInit = action === 'delete'
      ? { method: 'DELETE' }
      : action === 'run'
        ? { method: 'POST' }
        : {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: !schedule.enabled }),
          };
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`Unable to ${action} schedule`);
      if (action === 'run') {
        const run = await response.json();
        setNotice(`Run queued as job ${run.job_id}.`);
      } else {
        if (editingId === schedule.prompt_schedule_id) resetForm();
        setNotice(action === 'delete' ? 'Schedule deleted.' : 'Schedule updated.');
        await fetchSchedules();
      }
    } catch (mutationError) {
      setError(errorMessage(mutationError));
    }
  };

  return (
    <div className="schedules-page page-surface">
      <header className="page-header">
        <div>
          <p className="section-eyebrow">Automation</p>
          <h2>Schedules</h2>
          <p>Run saved prompts on a five-field cron schedule in the background.</p>
        </div>
      </header>

      {error && <div className="notice notice-error">{error}</div>}
      {notice && <div className="notice notice-success">{notice}</div>}

      <div className="schedules-layout">
        <form className="schedule-form" onSubmit={submitSchedule}>
          <div className="panel-title-row">
            <h3>{editingId === null ? 'New schedule' : 'Edit schedule'}</h3>
            {editingId !== null && (
              <button className="button button-secondary" type="button" onClick={resetForm}>
                Cancel
              </button>
            )}
          </div>

          <label>
            Name
            <input
              className="form-control"
              required
              maxLength={200}
              value={form.name}
              onChange={(event) => setForm({ ...form, name: event.target.value })}
              placeholder="Weekday briefing"
            />
          </label>
          <label>
            Prompt
            <textarea
              className="form-control schedule-prompt"
              required
              value={form.prompt}
              onChange={(event) => setForm({ ...form, prompt: event.target.value })}
              placeholder="Summarize my priorities for today."
            />
          </label>
          <div className="schedule-form-row">
            <label>
              Cron
              <input
                className="form-control schedule-monospace"
                required
                value={form.cron_expression}
                onChange={(event) => setForm({ ...form, cron_expression: event.target.value })}
              />
            </label>
            <label>
              Time zone
              <input
                className="form-control"
                required
                value={form.timezone}
                onChange={(event) => setForm({ ...form, timezone: event.target.value })}
              />
            </label>
          </div>
          <label>
            Runtime
            <select
              className="form-control"
              value={form.agent_type}
              onChange={(event) => setForm({
                ...form,
                agent_type: event.target.value as ScheduleForm['agent_type'],
              })}
            >
              <option value="">Use current default</option>
              <option value="local">Local</option>
              <option value="online">Online</option>
            </select>
          </label>
          <p className="schedule-help">Format: minute hour day-of-month month day-of-week</p>
          <button className="button" type="submit" disabled={saving}>
            {saving ? 'Saving…' : editingId === null ? 'Create schedule' : 'Save changes'}
          </button>
        </form>

        <section className="schedule-list-panel" aria-label="Configured cron tasks">
          <div className="panel-title-row">
            <h3>Configured tasks</h3>
            <span className="runtime-chip">{schedules.length}</span>
          </div>
          {loading ? (
            <div className="empty-state">Loading schedules…</div>
          ) : schedules.length === 0 ? (
            <div className="empty-state">No cron tasks configured yet.</div>
          ) : (
            <div className="schedule-list">
              {schedules.map((schedule) => (
                <article className="schedule-card" key={schedule.prompt_schedule_id}>
                  <div className="schedule-card-heading">
                    <div>
                      <h4>{schedule.name}</h4>
                      <code>{schedule.cron_expression}</code>
                      <span>{schedule.timezone}</span>
                    </div>
                    <span className={`status-badge ${schedule.enabled ? 'success' : 'warning'}`}>
                      {schedule.enabled ? 'Enabled' : 'Paused'}
                    </span>
                  </div>
                  <p className="schedule-prompt-preview">{schedule.prompt}</p>
                  <dl className="schedule-metadata">
                    <div><dt>Next run</dt><dd>{formatUtc(schedule.next_run_at)}</dd></div>
                    <div><dt>Last queued</dt><dd>{formatUtc(schedule.last_enqueued_at)}</dd></div>
                  </dl>
                  <div className="schedule-actions">
                    <button className="button button-secondary" type="button" onClick={() => editSchedule(schedule)}>Edit</button>
                    <button className="button button-secondary" type="button" onClick={() => void mutateSchedule(schedule, 'toggle')}>
                      {schedule.enabled ? 'Pause' : 'Enable'}
                    </button>
                    <button className="button" type="button" onClick={() => void mutateSchedule(schedule, 'run')}>Run now</button>
                    <button className="button button-danger" type="button" onClick={() => void mutateSchedule(schedule, 'delete')}>Delete</button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default Schedules;
