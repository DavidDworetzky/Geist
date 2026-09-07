import React from 'react';
import './App.css';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './AppShell';
import Chat from './Chat';
import WorkflowBuilder from './WorkflowBuilder';
import Files from './Files';
import Models from './Models';
import Settings from './Settings';
import Tools from './Tools';
import { BrandingProvider } from './branding';
import { UserSettingsProvider } from './Hooks/useUserSettings';
import { GeistPluginHost } from './plugins/runtime';
import './Motion.css';

function App() {
  return (
    <BrandingProvider>
      <UserSettingsProvider>
        <BrowserRouter>
          <GeistPluginHost />
          <AppShell>
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/chat/:chatId" element={<Chat />} />
              <Route path="/workflows" element={<WorkflowBuilder />} />
              <Route path="/workflows/:workflowId" element={<WorkflowBuilder />} />
              <Route path="/files" element={<Files />} />
              <Route path="/models" element={<Models />} />
              <Route path="/tools" element={<Tools />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </UserSettingsProvider>
    </BrandingProvider>
  );
}

export default App;
