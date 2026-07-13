import React, { useState, useCallback, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Connection,
  MarkerType,
  NodeTypes,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import useWorkflows, { WorkflowStep, WorkflowCreate, WorkflowUpdate } from './Hooks/useWorkflows';
import { NavLink, useParams, useNavigate } from 'react-router-dom';
import StagePanelIcon from './Components/StagePanelIcon';
import './WorkflowBuilder.css';

const WorkflowStepNode = ({ data, selected }: { data: any; selected: boolean }) => {
  const stepType = data.stepType || 'custom';

  return (
    <div className={`workflow-step-node workflow-step-${stepType} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Top} />
      <div className="step-header">
        <span className="step-type">{stepType.toUpperCase()}</span>
      </div>
      <div className="step-content">
        <h4>{data.label}</h4>
        {data.description && <p>{data.description}</p>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

const nodeTypes: NodeTypes = {
  workflowStep: WorkflowStepNode,
};

const WORKFLOW_PANEL_STORAGE_KEY = 'geist.workflowLibraryState';

type WorkflowPanelState = 'minimized' | 'expanded';

function getInitialWorkflowPanelState(): WorkflowPanelState {
  if (typeof window === 'undefined') {
    return 'minimized';
  }

  return window.localStorage.getItem(WORKFLOW_PANEL_STORAGE_KEY) === 'expanded' ? 'expanded' : 'minimized';
}

const WorkflowBuilder: React.FC = () => {
  const { workflowId } = useParams<{ workflowId?: string }>();
  const navigate = useNavigate();
  const { workflows, createWorkflow, updateWorkflow, getWorkflow, runWorkflow, runLoading, runResult, setRunResult } = useWorkflows();

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [workflowName, setWorkflowName] = useState('New Workflow');
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showNodeEditor, setShowNodeEditor] = useState(false);
  const [nodeFormData, setNodeFormData] = useState({
    label: '',
    description: '',
    stepType: 'custom' as WorkflowStep['step_type'],
    commandStr: '',
  });
  const [showRunDialog, setShowRunDialog] = useState(false);
  const [runInputData, setRunInputData] = useState('');
  const [workflowPanelState, setWorkflowPanelState] = useState<WorkflowPanelState>(getInitialWorkflowPanelState);
  const [workflowSearch, setWorkflowSearch] = useState('');
  const [editingWorkflowId, setEditingWorkflowId] = useState<number | null>(null);
  const [workflowTitleDraft, setWorkflowTitleDraft] = useState('');
  const [workflowTitleError, setWorkflowTitleError] = useState('');
  const [workflowTitleSaving, setWorkflowTitleSaving] = useState(false);

  const loadWorkflow = useCallback(async (id: number) => {
    const workflow = await getWorkflow(id);
    if (workflow) {
      setWorkflowName(workflow.name);

      const loadedNodes = workflow.steps.map((step, index) => ({
        id: `${step.step_id}`,
        type: 'workflowStep',
        position: { x: step.display_x || 100 + index * 200, y: step.display_y || 100 },
        data: {
          label: step.step_name,
          description: step.step_description,
          stepType: step.step_type,
          commandStr: step.command_str,
        },
      }));

      setNodes(loadedNodes);
    }
  }, [getWorkflow, setNodes]);

  useEffect(() => {
    if (workflowId) {
      void loadWorkflow(parseInt(workflowId, 10));
    }
  }, [workflowId, loadWorkflow]);

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds: Edge[]) => addEdge({
        ...params,
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
      }, eds));
    },
    [setEdges]
  );

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    setNodeFormData({
      label: node.data.label || '',
      description: node.data.description || '',
      stepType: node.data.stepType || 'custom',
      commandStr: node.data.commandStr || '',
    });
    setShowNodeEditor(true);
  }, []);

  const addNewNode = () => {
    const newNode: Node = {
      id: `node-${Date.now()}`,
      type: 'workflowStep',
      position: { x: 250, y: 250 },
      data: {
        label: 'New Step',
        description: '',
        stepType: 'custom',
        commandStr: '',
      },
    };
    setNodes((nds: Node[]) => nds.concat(newNode));
  };

  const updateNode = () => {
    if (selectedNode) {
      setNodes((nds: Node[]) =>
        nds.map((node: Node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ...nodeFormData,
              },
            };
          }
          return node;
        })
      );
      setShowNodeEditor(false);
      setSelectedNode(null);
    }
  };

  const deleteNode = () => {
    if (selectedNode) {
      setNodes((nds: Node[]) => nds.filter((node: Node) => node.id !== selectedNode.id));
      setEdges((eds: Edge[]) => eds.filter((edge: Edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id));
      setShowNodeEditor(false);
      setSelectedNode(null);
    }
  };

  const saveWorkflow = async () => {
    const steps: WorkflowStep[] = nodes.map((node: Node) => ({
      step_name: node.data.label,
      step_description: node.data.description,
      step_type: node.data.stepType,
      display_x: Math.round(node.position.x),
      display_y: Math.round(node.position.y),
      command_str: node.data.commandStr,
      step_status: 'pending',
    }));

    if (workflowId) {
      const update: WorkflowUpdate = {
        name: workflowName,
        steps,
      };
      const result = await updateWorkflow(parseInt(workflowId, 10), update);
      if (result) {
        alert('Workflow updated successfully.');
      }
    } else {
      const newWorkflow: WorkflowCreate = {
        name: workflowName,
        steps,
      };
      const result = await createWorkflow(newWorkflow);
      if (result) {
        alert('Workflow created successfully.');
        navigate(`/workflows/${result.workflow_id}`);
      }
    }
  };

  const handleRunWorkflow = async () => {
    if (!workflowId) {
      alert('Please save the workflow first.');
      return;
    }

    let inputData = {};
    if (runInputData.trim()) {
      try {
        inputData = JSON.parse(runInputData);
      } catch {
        alert('Invalid JSON input data.');
        return;
      }
    }

    const result = await runWorkflow(parseInt(workflowId, 10), inputData);
    if (result) {
      setShowRunDialog(false);
      setRunInputData('');
    }
  };

  const workflowLinks = workflows.map((workflow) => ({
    id: workflow.workflow_id,
    name: workflow.name,
    link: `/workflows/${workflow.workflow_id}`,
  }));
  const workflowSearchQuery = workflowSearch.trim().toLowerCase();
  const filteredWorkflowLinks = workflowSearchQuery
    ? workflowLinks.filter((workflow) => workflow.name.toLowerCase().includes(workflowSearchQuery))
    : workflowLinks;
  const workflowCountLabel = `${workflowLinks.length} ${workflowLinks.length === 1 ? 'workflow' : 'workflows'}`;

  const setWorkflowLibraryState = (nextState: WorkflowPanelState) => {
    setWorkflowPanelState(nextState);
    window.localStorage.setItem(WORKFLOW_PANEL_STORAGE_KEY, nextState);
    if (nextState === 'minimized') {
      setEditingWorkflowId(null);
      setWorkflowTitleDraft('');
      setWorkflowTitleError('');
    }
  };

  const startEditingWorkflowTitle = (workflow: typeof workflowLinks[number]) => {
    setEditingWorkflowId(workflow.id);
    setWorkflowTitleDraft(workflow.name);
    setWorkflowTitleError('');
  };

  const cancelEditingWorkflowTitle = () => {
    setEditingWorkflowId(null);
    setWorkflowTitleDraft('');
    setWorkflowTitleError('');
  };

  const saveWorkflowTitle = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (editingWorkflowId === null || workflowTitleSaving) return;

    const normalizedTitle = workflowTitleDraft.trim();
    if (!normalizedTitle) {
      setWorkflowTitleError('Enter a name for this workflow.');
      return;
    }

    setWorkflowTitleSaving(true);
    setWorkflowTitleError('');
    const result = await updateWorkflow(editingWorkflowId, { name: normalizedTitle });
    setWorkflowTitleSaving(false);

    if (!result) {
      setWorkflowTitleError('Unable to save this workflow name.');
      return;
    }

    if (workflowId && parseInt(workflowId, 10) === editingWorkflowId) {
      setWorkflowName(result.name);
    }
    cancelEditingWorkflowTitle();
  };

  const handleNewWorkflow = () => {
    setNodes([]);
    setEdges([]);
    setWorkflowName('New Workflow');
    setWorkflowLibraryState('minimized');
    navigate('/workflows');
  };

  return (
    <div className={`WorkflowBuilderContainer workflow-library-${workflowPanelState}`}>
      <main className="WorkflowContent">
        <div className="workflow-stage">
          <div className="workflow-canvas" aria-hidden={workflowPanelState === 'expanded'}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              fitView
            >
              <Background />
              <Controls />
              <MiniMap />
            </ReactFlow>
            {nodes.length === 0 && workflowPanelState === 'minimized' && (
              <div className="workflow-canvas-empty">Start building a workflow.</div>
            )}
          </div>

          {workflowPanelState === 'minimized' && (
            <div className="workflow-stage-actions" role="toolbar" aria-label="Workflow actions">
              <button onClick={addNewNode} className="button" type="button" aria-label="Add step" title="Add step">
                <StagePanelIcon name="plus" />
                <span>Add Step</span>
              </button>
              <button onClick={saveWorkflow} className="button" type="button" aria-label="Save workflow" title="Save workflow">
                <StagePanelIcon name="save" />
                <span>Save</span>
              </button>
              {workflowId && (
                <button
                  onClick={() => setShowRunDialog(true)}
                  className="button"
                  type="button"
                  aria-label="Run workflow"
                  title="Run workflow"
                  disabled={runLoading}
                >
                  <StagePanelIcon name="play" />
                  <span>{runLoading ? 'Running...' : 'Run'}</span>
                </button>
              )}
            </div>
          )}

          <aside
            className={`WorkflowLibrary stage-panel workflow-library-panel-host stage-panel-${workflowPanelState}`}
            aria-label="Workflow library"
            data-state={workflowPanelState}
          >
            <div className="stage-panel-surface">
              {workflowPanelState === 'minimized' ? (
                <div className="workflow-minimized-controls stage-panel-compact-controls" aria-label="Workflow shortcuts">
                  <button className="button workflow-minimized-new-button stage-panel-primary-button" type="button" onClick={handleNewWorkflow} aria-label="New workflow" title="New workflow">
                    <span>New Workflow</span>
                    <StagePanelIcon name="plus" />
                  </button>
                  <button
                    className="workflow-minimized-open-button stage-panel-open-button"
                    type="button"
                    onClick={() => setWorkflowLibraryState('expanded')}
                    aria-label="Expand workflow library"
                    aria-expanded={false}
                    title="Expand workflow library"
                  >
                    <StagePanelIcon name="maximize" />
                  </button>
                </div>
              ) : (
                <div className="workflow-library-panel stage-panel-content">
                  <div className="stage-panel-content-column">
                    <header className="workflow-library-header stage-panel-header">
                      <div>
                        <p className="section-eyebrow">Library</p>
                        <h2>Workflows</h2>
                      </div>
                      <button className="icon-action" type="button" onClick={() => setWorkflowLibraryState('minimized')} aria-label="Close workflow library" title="Close workflow library">
                        <StagePanelIcon name="close" />
                      </button>
                    </header>

                    <button className="button workflow-new-button" type="button" onClick={handleNewWorkflow}>
                      <StagePanelIcon name="plus" />
                      <span>New Workflow</span>
                    </button>

                    <label className="workflow-search" htmlFor="workflow-library-search">
                      <StagePanelIcon name="search" />
                      <input
                        id="workflow-library-search"
                        type="search"
                        value={workflowSearch}
                        onChange={(event) => setWorkflowSearch(event.target.value)}
                        placeholder="Search workflows"
                        autoFocus
                      />
                    </label>

                    <section className="workflow-library-summary" aria-label="Saved workflow count">
                      <span className="workflow-library-icon">
                        <StagePanelIcon name="workflow" />
                      </span>
                      <span>
                        <strong>Saved workflows</strong>
                        <small>{workflowCountLabel}</small>
                      </span>
                    </section>

                    <div className="workflow-library-section-title">
                      <span>Workflow library</span>
                      <span>{workflowCountLabel}</span>
                    </div>

                    {filteredWorkflowLinks.length > 0 ? (
                      <div className="workflow-session-list chat-session-list" role="list" aria-label="Saved workflows">
                        {filteredWorkflowLinks.map((workflow) => {
                          const isEditing = editingWorkflowId === workflow.id;
                          return (
                            <div
                              className={`workflow-session-row chat-session-row${isEditing ? ' editing' : ''}`}
                              role="listitem"
                              key={workflow.id}
                            >
                              {isEditing ? (
                                <form className="workflow-title-edit-form chat-title-edit-form" onSubmit={(event) => void saveWorkflowTitle(event)}>
                                  <input
                                    className="workflow-title-input chat-title-input"
                                    aria-label="Workflow name"
                                    value={workflowTitleDraft}
                                    onChange={(event) => setWorkflowTitleDraft(event.target.value)}
                                    onKeyDown={(event) => {
                                      if (event.key === 'Escape') {
                                        event.preventDefault();
                                        cancelEditingWorkflowTitle();
                                      }
                                    }}
                                    maxLength={120}
                                    disabled={workflowTitleSaving}
                                    autoFocus
                                  />
                                  <button className="icon-action" type="submit" aria-label="Save workflow name" title="Save workflow name" disabled={workflowTitleSaving}>
                                    <StagePanelIcon name="check" />
                                  </button>
                                  <button className="icon-action" type="button" onClick={cancelEditingWorkflowTitle} aria-label="Cancel renaming" title="Cancel" disabled={workflowTitleSaving}>
                                    <StagePanelIcon name="close" />
                                  </button>
                                  {workflowTitleError && <span className="chat-title-error" role="alert">{workflowTitleError}</span>}
                                </form>
                              ) : (
                                <>
                                  <NavLink
                                    to={workflow.link}
                                    className="list-link workflow-session-link chat-session-link"
                                    onClick={() => setWorkflowLibraryState('minimized')}
                                  >
                                    <span className="chat-history-item">{workflow.name}</span>
                                  </NavLink>
                                  <button
                                    className="workflow-session-edit-button chat-session-edit-button"
                                    type="button"
                                    onClick={() => startEditingWorkflowTitle(workflow)}
                                    aria-label={`Rename ${workflow.name}`}
                                    title="Rename workflow"
                                  >
                                    <StagePanelIcon name="edit" />
                                  </button>
                                </>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="workflow-library-empty">
                        <strong>{workflowSearchQuery ? 'No matching workflows' : 'No workflows yet'}</strong>
                        <span>{workflowSearchQuery ? 'Try a different search.' : 'Create a workflow to build a reusable local automation.'}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>

        {showNodeEditor && (
          <div className="node-editor-overlay">
            <div className="node-editor">
              <h3>Edit Step</h3>
              <div className="form-group">
                <label>Step Name</label>
                <input
                  className="form-control"
                  type="text"
                  value={nodeFormData.label}
                  onChange={(e) => setNodeFormData({ ...nodeFormData, label: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  className="form-control"
                  value={nodeFormData.description}
                  onChange={(e) => setNodeFormData({ ...nodeFormData, description: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Step Type</label>
                <select
                  className="form-control"
                  value={nodeFormData.stepType}
                  onChange={(e) => setNodeFormData({ ...nodeFormData, stepType: e.target.value as WorkflowStep['step_type'] })}
                >
                  <option value="trigger">Trigger (Input)</option>
                  <option value="map">Map</option>
                  <option value="filter">Filter</option>
                  <option value="reduce">Reduce</option>
                  <option value="expand">Expand</option>
                  <option value="custom">Custom</option>
                  <option value="llm">LLM</option>
                  <option value="agent">Agent</option>
                  <option value="adapter">Adapter (Output)</option>
                </select>
              </div>
              <div className="form-group">
                <label>Command/Configuration</label>
                <textarea
                  className="form-control"
                  value={nodeFormData.commandStr}
                  onChange={(e) => setNodeFormData({ ...nodeFormData, commandStr: e.target.value })}
                  placeholder="Enter command or configuration..."
                />
              </div>
              <div className="editor-actions">
                <button onClick={updateNode} className="button">Update</button>
                <button onClick={deleteNode} className="button button-danger">Delete</button>
                <button onClick={() => setShowNodeEditor(false)} className="button button-secondary">Cancel</button>
              </div>
            </div>
          </div>
        )}

        {showRunDialog && (
          <div className="run-dialog-overlay">
            <div className="run-dialog">
              <h3>Run Workflow</h3>
              <div className="form-group">
                <label>Input Data (JSON)</label>
                <textarea
                  className="form-control"
                  value={runInputData}
                  onChange={(e) => setRunInputData(e.target.value)}
                  placeholder='{"key": "value"}'
                  rows={6}
                />
                <small>Enter JSON data to pass to the workflow (optional).</small>
              </div>
              <div className="dialog-actions">
                <button onClick={handleRunWorkflow} className="button" disabled={runLoading}>
                  {runLoading ? 'Running...' : 'Run'}
                </button>
                <button onClick={() => setShowRunDialog(false)} className="button button-secondary" disabled={runLoading}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {runResult && (
          <div className="run-results">
            <h3>Workflow Execution Result</h3>
            <div className="result-item">
              <strong>Status:</strong>
              <span className={`status ${runResult.status}`}>{runResult.status}</span>
            </div>
            <div className="result-item">
              <strong>Run ID:</strong> {runResult.run_id}
            </div>
            <div className="result-item">
              <strong>Started:</strong> {runResult.started_at ? new Date(runResult.started_at).toLocaleString() : 'N/A'}
            </div>
            <div className="result-item">
              <strong>Completed:</strong> {runResult.completed_at ? new Date(runResult.completed_at).toLocaleString() : 'N/A'}
            </div>
            {runResult.error_message && (
              <div className="result-item error">
                <strong>Error:</strong> {runResult.error_message}
              </div>
            )}
            <div className="result-item">
              <strong>Output Data:</strong>
              <pre>{JSON.stringify(runResult.output_data, null, 2)}</pre>
            </div>
            <button onClick={() => setRunResult(null)} className="button button-secondary">
              Close Results
            </button>
          </div>
        )}
      </main>
    </div>
  );
};

export default WorkflowBuilder;
