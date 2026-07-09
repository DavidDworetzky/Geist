# Settings UI Implementation Plan

## Overview
Implement a comprehensive Settings UI for the Geist application to allow users to manage their preferences, agent configurations, and default behaviors through a user-friendly interface.

## Current State Analysis

### Backend (✅ Complete)
- **Database Model**: `app/models/database/user_settings.py` - SQLAlchemy model with all required fields
- **Service Layer**: `app/services/user_settings_service.py` - Complete CRUD operations
- **API Models**: `app/models/user_settings.py` - Pydantic models for validation
- **API Endpoints**: `app/api/v1/endpoints/user_settings.py` - Full REST API
  - `GET /api/v1/user-settings/` - Get current user settings
  - `PUT /api/v1/user-settings/` - Update settings
  - `POST /api/v1/user-settings/reset` - Reset to defaults
  - `GET /api/v1/user-settings/agent-config/preview` - Preview agent config
- **Routing**: Already registered in `app/main.py`

### Frontend (❌ Missing)
- Settings page component
- Route configuration
- Navigation link
- Custom hooks for API integration
- Form components for different settings sections
- Tests

## Implementation Plan

### Phase 1: API Integration Layer
**Files to Create:**

#### 1.1 Custom Hook for Settings Management
**File**: `client/geist/src/Hooks/useUserSettings.tsx`

```typescript
import { useState, useEffect, useCallback } from 'react';

export interface UserSettings {
  user_settings_id: number;
  user_id: number;
  default_agent_type: string;
  default_local_model: string;
  default_online_model: string;
  default_online_provider: string;
  default_file_archives: number[];
  enable_rag_by_default: boolean;
  default_max_tokens: number;
  default_temperature: number;
  default_top_p: number;
  default_frequency_penalty: number;
  default_presence_penalty: number;
  backup_providers: BackupProvider[];
  ui_preferences: Record<string, any>;
  create_date: string;
  update_date: string;
}

export interface BackupProvider {
  name: string;
  base_url: string;
  model: string;
  api_key?: string;
  priority: number;
}

export interface UserSettingsUpdate {
  default_agent_type?: string;
  default_local_model?: string;
  default_online_model?: string;
  default_online_provider?: string;
  default_file_archives?: number[];
  enable_rag_by_default?: boolean;
  default_max_tokens?: number;
  default_temperature?: number;
  default_top_p?: number;
  default_frequency_penalty?: number;
  default_presence_penalty?: number;
  backup_providers?: BackupProvider[];
  ui_preferences?: Record<string, any>;
}
```

**Features:**
- `useUserSettings()` - Main hook for fetching and managing settings
- `fetchSettings()` - Get current settings
- `updateSettings(updates)` - Update specific fields
- Loading and error states
- Auto-refresh after updates

### Phase 2: UI Components

#### 2.1 Main Settings Page
**File**: `client/geist/src/Settings.tsx`

**Structure:**
- Tab-based layout with sections:
  1. **Agent Configuration** - Model selection, agent type
  2. **Generation Parameters** - Temperature, tokens, penalties
  3. **RAG & Files** - File archives, RAG settings
  4. **UI Preferences** - Theme, display options

**Key Features:**
- Live preview of changes (unsaved state indicator)
- Form validation
- Save/Cancel buttons.
- Success/Error notifications
- Responsive design matching existing patterns

#### 2.2 Settings Sub-Components
**Files to Create:**

**a) `client/geist/src/Components/AgentConfigSection.tsx`**
- Agent type selector (local/online)
- Model selection dropdowns
- Conditional rendering based on agent type
- Provider selection for online agents

**b) `client/geist/src/Components/GenerationParamsSection.tsx`**
- Slider controls for:
  - Temperature (0.0 - 2.0)
  - Top P (0.0 - 1.0)
  - Max Tokens (1 - 4096+)
  - Frequency Penalty (0.0 - 2.0)
  - Presence Penalty (0.0 - 2.0)
- Real-time value display
- Reset to defaults per parameter

**c) `client/geist/src/Components/RAGSettingsSection.tsx`**
- Toggle for RAG by default
- Multi-select for default file archives
- Integration with existing file upload system
- Clear/Select all functionality

**d) `client/geist/src/Components/BackupProvidersSection.tsx`**
- List of backup providers
- Add/Edit/Delete provider modal
- Priority ordering (drag-and-drop or up/down buttons)
- Form fields:
  - Provider name
  - Base URL
  - Model
  - API key (masked input)
  - Priority

### Phase 3: Navigation & Routing

#### 3.1 Add Settings Route
**File**: `client/geist/src/App.tsx`

**Changes:**
```typescript
import Settings from './Settings';

// Add to routes:
<Route path="/settings" element={<Settings/>}/>
```

#### 3.2 Add Navigation Link
**File**: `client/geist/src/App.tsx`

**Changes:**
```typescript
const links = [
  // ... existing links
  {
    name: 'Settings',
    link: '/settings',
    svg: 'M495.9 166.6c3.2 8.7 .5 18.4-6.4 24.6l-43.3 39.4c1.1 8.3 1.7 16.8 1.7 25.4s-.6 17.1-1.7 25.4l43.3 39.4c6.9 6.2 9.6 15.9 6.4 24.6c-4.4 11.9-9.7 23.3-15.8 34.3l-4.7 8.1c-6.6 11-14 21.4-22.1 31.2c-5.9 7.2-15.7 9.6-24.5 6.8l-55.7-17.7c-13.4 10.3-28.2 18.9-44 25.4l-12.5 57.1c-2 9.1-9 16.3-18.2 17.8c-13.8 2.3-28 3.5-42.5 3.5s-28.7-1.2-42.5-3.5c-9.2-1.5-16.2-8.7-18.2-17.8l-12.5-57.1c-15.8-6.5-30.6-15.1-44-25.4L83.1 425.9c-8.8 2.8-18.6 .3-24.5-6.8c-8.1-9.8-15.5-20.2-22.1-31.2l-4.7-8.1c-6.1-11-11.4-22.4-15.8-34.3c-3.2-8.7-.5-18.4 6.4-24.6l43.3-39.4C64.6 273.1 64 264.6 64 256s.6-17.1 1.7-25.4L22.4 191.2c-6.9-6.2-9.6-15.9-6.4-24.6c4.4-11.9 9.7-23.3 15.8-34.3l4.7-8.1c6.6-11 14-21.4 22.1-31.2c5.9-7.2 15.7-9.6 24.5-6.8l55.7 17.7c13.4-10.3 28.2-18.9 44-25.4l12.5-57.1c2-9.1 9-16.3 18.2-17.8C227.3 1.2 241.5 0 256 0s28.7 1.2 42.5 3.5c9.2 1.5 16.2 8.7 18.2 17.8l12.5 57.1c15.8 6.5 30.6 15.1 44 25.4l55.7-17.7c8.8-2.8 18.6-.3 24.5 6.8c8.1 9.8 15.5 20.2 22.1 31.2l4.7 8.1c6.1 11 11.4 22.4 15.8 34.3zM256 336a80 80 0 1 0 0-160 80 80 0 1 0 0 160z'
  }
];
```

### Phase 4: Styling & UX

#### 4.1 Settings-Specific Styles
**File**: `client/geist/src/Settings.css`

**Features:**
- Consistent with existing app design (reference `App.css`)
- Card-based layout for sections
- Responsive grid/flexbox
- Smooth transitions
- Form control styling
- Tab navigation styling
- Modal styling for backup providers

#### 4.2 Form Controls
Reusable components:
- Slider with value display
- Toggle switch
- Select dropdown
- Multi-select with chips
- Masked input (for API keys)
- Number input with +/- buttons

### Phase 5: Advanced Features

#### 5.1 Validation & Feedback
- Real-time validation for:
  - URL formats for backup providers
  - Numeric ranges for generation params
  - Required fields
- Inline error messages
- Success notifications (toast/banner)
- Unsaved changes warning

#### 5.2 Preview Functionality
**Integration with Backend Preview Endpoint:**
- Show computed agent configuration
- Display which settings will be used

## Implementation Checklist

### Backend (Already Complete ✅)
- [x] Database model
- [x] Service layer
- [x] API endpoints
- [x] Pydantic models
- [x] Router registration

### Frontend (To Implement)

#### Phase 1: Foundation
- [ ] Create `useUserSettings.tsx` hook
- [ ] Define TypeScript interfaces
- [ ] Test API integration

#### Phase 2: Core Components
- [ ] Create `Settings.tsx` main page
- [ ] Create `AgentConfigSection.tsx`
- [ ] Create `GenerationParamsSection.tsx`
- [ ] Create `RAGSettingsSection.tsx`
- [ ] Create `UIPreferencesSection.tsx`

#### Phase 3: Reusable Controls
- [ ] Create slider component
- [ ] Create toggle component
- [ ] Create multi-select component
- [ ] Create masked input component
- [ ] Create modal component (if not exists)

#### Phase 4: Integration
- [ ] Add route to `App.tsx`
- [ ] Add navigation link
- [ ] Create `Settings.css`

#### Phase 5: Polish
- [ ] Add form validation
- [ ] Add success/error notifications
- [ ] Add unsaved changes warning
- [ ] Add basic loading states
- [ ] Add responsive design

#### Phase 6: Advanced Features
- [ ] Implement preview functionality
- [ ] Add tooltips/help text
- [ ] Add keyboard shortcuts

## Technical Decisions

### State Management
- Use React hooks (useState, useEffect)
- Custom hook for API integration
- Local component state for form inputs
- No global state library needed (settings are user-specific)

### Styling Approach
- Inline styles (matching existing Files.tsx pattern)
- Minimal CSS file for complex layouts
- Consistent with existing design language
- Mobile-first responsive design

### Form Handling
- Controlled components
- Debounced validation
- Optimistic UI updates with rollback on error
- Save on blur or explicit save button

### Error Handling
- Try-catch blocks for API calls
- User-friendly error messages
- Graceful degradation
- Retry mechanisms for failed requests

## Testing Strategy

### Unit Tests
- Test custom hooks with mock API responses
- Test form validation logic
- Test component rendering with different props

## Accessibility Considerations
- Proper ARIA labels for form controls
- Keyboard navigation support
- Focus management for modals
- Screen reader friendly
- Color contrast compliance

## Performance Optimization
- Lazy load settings page
- Debounce API calls during typing


## Success Criteria
- [ ] User can view all settings in organized UI
- [ ] User can update any setting and see changes persist
- [ ] User can reset to defaults
- [ ] Form validation prevents invalid input
- [ ] UI is responsive and accessible
- [ ] Changes are reflected immediately in agent behavior
- [ ] Preview shows correct agent configuration
- [ ] Backup providers can be managed easily
- [ ] No console errors or warnings
- [ ] All existing tests still pass


## UI Mockup Structure

```
┌─────────────────────────────────────────────────────┐
│ Settings                                            │
├─────────────────────────────────────────────────────┤
│ [Agent Config] [Generation] [RAG] [Providers] [UI] │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Agent Configuration                                │
│  ┌───────────────────────────────────────────┐     │
│  │ Agent Type:  [Local ▼] [Online ▼]        │     │
│  │                                           │     │
│  │ Local Model: [Meta-Llama-3.1-8B ▼]      │     │
│  │ OR                                        │     │
│  │ Online Provider: [OpenAI ▼]              │     │
│  │ Online Model: [gpt-4 ▼]                 │     │
│  └───────────────────────────────────────────┘     │
│                                                     │
│  Generation Parameters                              │
│  ┌───────────────────────────────────────────┐     │
│  │ Temperature:    [━━━●━━━━] 0.7          │     │
│  │ Max Tokens:     [━━━━━●━━] 500          │     │
│  │ Top P:          [━━━━━━━●] 0.9          │     │
│  │ Frequency Pen:  [●━━━━━━━] 0.0          │     │
│  │ Presence Pen:   [●━━━━━━━] 0.0          │     │
│  └───────────────────────────────────────────┘     │
│                                                     │
│  [Save] [Cancel] [Reset to Defaults]               │
└─────────────────────────────────────────────────────┘
```

## File Structure Summary

```
client/geist/src/
├── Settings.tsx                         # Main settings page
├── Settings.css                         # Settings styles
├── Hooks/
│   └── useUserSettings.tsx             # Settings API hook
└── Components/
    ├── AgentConfigSection.tsx          # Agent config form
    ├── GenerationParamsSection.tsx     # Generation params form
    ├── RAGSettingsSection.tsx          # RAG settings form
    ├── BackupProvidersSection.tsx      # Backup providers management
    ├── UIPreferencesSection.tsx        # UI preferences
    ├── SettingsSlider.tsx              # Reusable slider
    ├── SettingsToggle.tsx              # Reusable toggle
    └── BackupProviderModal.tsx         # Add/edit provider modal
```

## API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/user-settings/` | Get current user settings |
| PUT | `/api/v1/user-settings/` | Update settings |
| POST | `/api/v1/user-settings/reset` | Reset to defaults |
| GET | `/api/v1/user-settings/agent-config/preview` | Preview agent config |

## Conclusion

This plan provides a comprehensive roadmap for implementing a fully-featured Settings UI that integrates seamlessly with the existing Geist application. The implementation will follow existing patterns from the Files page while introducing new reusable components for form controls. The settings will provide users with complete control over their agent configurations, generation parameters, and application preferences.

