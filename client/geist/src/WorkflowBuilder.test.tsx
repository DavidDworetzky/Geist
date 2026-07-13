import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import WorkflowBuilder from './WorkflowBuilder';

jest.mock('reactflow', () => {
  const React = require('react');

  return {
    __esModule: true,
    default: ({ children }: { children: any }) => React.createElement('div', { 'data-testid': 'workflow-canvas' }, children),
    Background: () => null,
    Controls: () => null,
    MiniMap: () => null,
    Handle: () => null,
    Position: { Top: 'top', Bottom: 'bottom' },
    MarkerType: { ArrowClosed: 'arrowclosed' },
    addEdge: (connection: unknown, edges: unknown[]) => [...edges, connection],
    useNodesState: (initial: unknown[]) => {
      const [nodes, setNodes] = React.useState(initial);
      return [nodes, setNodes, jest.fn()];
    },
    useEdgesState: (initial: unknown[]) => {
      const [edges, setEdges] = React.useState(initial);
      return [edges, setEdges, jest.fn()];
    },
  };
});

const mockUpdateWorkflow = jest.fn();

jest.mock('./Hooks/useWorkflows', () => ({
  __esModule: true,
  default: () => ({
    workflows: [
      {
        workflow_id: 7,
        user_id: 1,
        name: 'Morning prep',
        steps: [],
      },
    ],
    createWorkflow: jest.fn(),
    updateWorkflow: mockUpdateWorkflow,
    getWorkflow: jest.fn(),
    runWorkflow: jest.fn(),
    runLoading: false,
    runResult: null,
    setRunResult: jest.fn(),
  }),
}));

describe('Workflow library panel', () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockUpdateWorkflow.mockReset();
    mockUpdateWorkflow.mockResolvedValue({
      workflow_id: 7,
      user_id: 1,
      name: 'Morning routine',
      steps: [],
    });
  });

  it('uses the same compact-to-stage panel interaction and inline renaming as Chat', async () => {
    render(
      <MemoryRouter initialEntries={['/workflows']}>
        <WorkflowBuilder />
      </MemoryRouter>
    );

    const library = screen.getByRole('complementary', { name: 'Workflow library' });
    expect(library).toHaveAttribute('data-state', 'minimized');
    expect(library).toHaveClass('stage-panel-minimized');
    const workflowStage = library.closest('.workflow-stage');
    expect(workflowStage).not.toBeNull();
    expect(within(library).getByRole('button', { name: 'New workflow' })).toBeInTheDocument();
    expect(screen.getByText('Start building a workflow.')).toBeInTheDocument();

    const actions = within(workflowStage as HTMLElement).getByRole('toolbar', { name: 'Workflow actions' });
    expect(within(actions).getByRole('button', { name: 'Add step' })).toBeInTheDocument();
    expect(within(actions).getByRole('button', { name: 'Save workflow' })).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Workflow name' })).not.toBeInTheDocument();

    fireEvent.click(within(library).getByRole('button', { name: 'Expand workflow library' }));

    expect(library).toHaveAttribute('data-state', 'expanded');
    expect(library).toHaveClass('stage-panel-expanded');
    expect(screen.queryByRole('toolbar', { name: 'Workflow actions' })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Workflows' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Morning prep' })).toBeInTheDocument();
    expect(window.localStorage.getItem('geist.workflowLibraryState')).toBe('expanded');

    const search = within(library).getByRole('searchbox');
    fireEvent.change(search, { target: { value: 'missing' } });
    expect(screen.queryByRole('link', { name: 'Morning prep' })).not.toBeInTheDocument();
    expect(screen.getByText('No matching workflows')).toBeInTheDocument();

    fireEvent.change(search, { target: { value: 'morning' } });

    fireEvent.click(screen.getByRole('button', { name: 'Rename Morning prep' }));
    const nameInput = screen.getByRole('textbox', { name: 'Workflow name' });
    fireEvent.change(nameInput, { target: { value: '  Morning routine  ' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Save workflow name' }));
    });

    expect(mockUpdateWorkflow).toHaveBeenCalledWith(7, { name: 'Morning routine' });
    expect(screen.queryByRole('textbox', { name: 'Workflow name' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('link', { name: 'Morning prep' }));

    expect(library).toHaveAttribute('data-state', 'minimized');
    expect(screen.queryByRole('heading', { name: 'Workflows' })).not.toBeInTheDocument();
    expect(window.localStorage.getItem('geist.workflowLibraryState')).toBe('minimized');
  });
});
