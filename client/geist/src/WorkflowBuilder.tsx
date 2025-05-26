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
import { useParams, useNavigate } from 'react-router-dom';
import LinkList, { ListItem } from './Components/LinkList';
import './WorkflowBuilder.css';

// Custom node component for workflow steps
const WorkflowStepNode = ({ data, selected }: { data: any; selected: boolean }) => {
    const stepTypeColors: Record<string, string> = {
        map: '#ff6b6b',
        filter: '#4ecdc4',
        reduce: '#45b7d1',
        expand: '#96ceb4',
        custom: '#dda0dd',
        llm: '#ffd93d',
        agent: '#ff8b94',
    };

    const bgColor = stepTypeColors[data.stepType] || '#ffffff';

    return (
        <div 
            className={`workflow-step-node ${selected ? 'selected' : ''}`}
            style={{ backgroundColor: bgColor }}
        >
            <Handle type="target" position={Position.Top} />
            <div className="step-header">
                <span className="step-type">{data.stepType.toUpperCase()}</span>
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

const WorkflowBuilder: React.FC = () => {
    const { workflowId } = useParams<{ workflowId?: string }>();
    const navigate = useNavigate();
    const { workflows, createWorkflow, updateWorkflow, getWorkflow } = useWorkflows();
    
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

    const loadWorkflow = useCallback(async (id: number) => {
        const workflow = await getWorkflow(id);
        if (workflow) {
            setWorkflowName(workflow.name);
            
            // Convert workflow steps to nodes
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
            // TODO: Load edges from workflow connections when backend supports it
        }
    }, [getWorkflow, setNodes]);

    // Load workflow if editing existing one
    useEffect(() => {
        if (workflowId) {
            loadWorkflow(parseInt(workflowId));
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

    const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
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
            // Update existing workflow
            const update: WorkflowUpdate = {
                name: workflowName,
                steps,
            };
            const result = await updateWorkflow(parseInt(workflowId), update);
            if (result) {
                alert('Workflow updated successfully!');
            }
        } else {
            // Create new workflow
            const newWorkflow: WorkflowCreate = {
                name: workflowName,
                steps,
            };
            const result = await createWorkflow(newWorkflow);
            if (result) {
                alert('Workflow created successfully!');
                navigate(`/workflows/${result.workflow_id}`);
            }
        }
    };

    // Create workflow links for sidebar
    const workflowLinks: ListItem[] = workflows.map((workflow) => ({
        name: workflow.name,
        link: `/workflows/${workflow.workflow_id}`,
        date: new Date(), // You might want to add created_at to the workflow model
    }));

    return (
        <div className="WorkflowBuilderContainer">
            <div className="WorkflowSidebar">
                <div className="sidebar-header">
                    <h3>Workflows</h3>
                    <button onClick={() => navigate('/workflows')} className="new-workflow-btn">
                        + New Workflow
                    </button>
                </div>
                <LinkList listItems={workflowLinks} />
            </div>
            
            <div className="WorkflowContent">
                <div className="workflow-header">
                    <input
                        type="text"
                        value={workflowName}
                        onChange={(e) => setWorkflowName(e.target.value)}
                        className="workflow-name-input"
                        placeholder="Workflow Name"
                    />
                    <div className="workflow-actions">
                        <button onClick={addNewNode} className="add-node-btn">
                            + Add Step
                        </button>
                        <button onClick={saveWorkflow} className="save-workflow-btn">
                            Save Workflow
                        </button>
                    </div>
                </div>
                
                <div className="workflow-canvas">
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
                </div>

                {showNodeEditor && (
                    <div className="node-editor-overlay">
                        <div className="node-editor">
                            <h3>Edit Step</h3>
                            <div className="form-group">
                                <label>Step Name:</label>
                                <input
                                    type="text"
                                    value={nodeFormData.label}
                                    onChange={(e) => setNodeFormData({ ...nodeFormData, label: e.target.value })}
                                />
                            </div>
                            <div className="form-group">
                                <label>Description:</label>
                                <textarea
                                    value={nodeFormData.description}
                                    onChange={(e) => setNodeFormData({ ...nodeFormData, description: e.target.value })}
                                />
                            </div>
                            <div className="form-group">
                                <label>Step Type:</label>
                                <select
                                    value={nodeFormData.stepType}
                                    onChange={(e) => setNodeFormData({ ...nodeFormData, stepType: e.target.value as WorkflowStep['step_type'] })}
                                >
                                    <option value="map">Map</option>
                                    <option value="filter">Filter</option>
                                    <option value="reduce">Reduce</option>
                                    <option value="expand">Expand</option>
                                    <option value="custom">Custom</option>
                                    <option value="llm">LLM</option>
                                    <option value="agent">Agent</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label>Command/Configuration:</label>
                                <textarea
                                    value={nodeFormData.commandStr}
                                    onChange={(e) => setNodeFormData({ ...nodeFormData, commandStr: e.target.value })}
                                    placeholder="Enter command or configuration..."
                                />
                            </div>
                            <div className="editor-actions">
                                <button onClick={updateNode} className="update-btn">Update</button>
                                <button onClick={deleteNode} className="delete-btn">Delete</button>
                                <button onClick={() => setShowNodeEditor(false)} className="cancel-btn">Cancel</button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default WorkflowBuilder; 