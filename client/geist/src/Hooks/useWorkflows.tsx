import { useState, useEffect } from 'react';
import axios from 'axios';

export interface WorkflowStep {
    step_id?: number;
    workflow_id?: number;
    step_name: string;
    step_description?: string;
    step_status?: string;
    display_x?: number;
    display_y?: number;
    command_str?: string;
    step_type: 'map' | 'filter' | 'reduce' | 'expand' | 'custom' | 'llm' | 'agent';
}

export interface Workflow {
    workflow_id: number;
    user_id: number;
    name: string;
    steps: WorkflowStep[];
}

export interface WorkflowCreate {
    name: string;
    steps?: WorkflowStep[];
}

export interface WorkflowUpdate {
    name?: string;
    steps?: WorkflowStep[];
}

const useWorkflows = () => {
    const [workflows, setWorkflows] = useState<Workflow[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const fetchWorkflows = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get<Workflow[]>('/api/v1/workflows/');
            setWorkflows(response.data);
        } catch (err) {
            setError('Failed to fetch workflows');
            console.error('Error fetching workflows:', err);
        } finally {
            setLoading(false);
        }
    };

    const createWorkflow = async (workflow: WorkflowCreate): Promise<Workflow | null> => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.post<Workflow>('/api/v1/workflows/', workflow);
            await fetchWorkflows(); // Refresh the list
            return response.data;
        } catch (err) {
            setError('Failed to create workflow');
            console.error('Error creating workflow:', err);
            return null;
        } finally {
            setLoading(false);
        }
    };

    const updateWorkflow = async (workflowId: number, workflow: WorkflowUpdate): Promise<Workflow | null> => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.put<Workflow>(`/api/v1/workflows/${workflowId}`, workflow);
            await fetchWorkflows(); // Refresh the list
            return response.data;
        } catch (err) {
            setError('Failed to update workflow');
            console.error('Error updating workflow:', err);
            return null;
        } finally {
            setLoading(false);
        }
    };

    const deleteWorkflow = async (workflowId: number): Promise<boolean> => {
        setLoading(true);
        setError(null);
        try {
            await axios.delete(`/api/v1/workflows/${workflowId}`);
            await fetchWorkflows(); // Refresh the list
            return true;
        } catch (err) {
            setError('Failed to delete workflow');
            console.error('Error deleting workflow:', err);
            return false;
        } finally {
            setLoading(false);
        }
    };

    const getWorkflow = async (workflowId: number): Promise<Workflow | null> => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get<Workflow>(`/api/v1/workflows/${workflowId}`);
            return response.data;
        } catch (err) {
            setError('Failed to fetch workflow');
            console.error('Error fetching workflow:', err);
            return null;
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchWorkflows();
    }, []);

    return {
        workflows,
        loading,
        error,
        createWorkflow,
        updateWorkflow,
        deleteWorkflow,
        getWorkflow,
        fetchWorkflows
    };
};

export default useWorkflows; 