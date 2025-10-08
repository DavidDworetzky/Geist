import React, { useState } from 'react';
import { BackupProvider } from '../Hooks/useUserSettings';

interface BackupProvidersSectionProps {
  providers: BackupProvider[];
  onProvidersChange: (providers: BackupProvider[]) => void;
}

const BackupProvidersSection: React.FC<BackupProvidersSectionProps> = ({
  providers,
  onProvidersChange
}) => {
  const [showModal, setShowModal] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [formData, setFormData] = useState<BackupProvider>({
    name: '',
    base_url: '',
    model: '',
    api_key: '',
    priority: 1
  });

  const openAddModal = () => {
    setEditingIndex(null);
    setFormData({ name: '', base_url: '', model: '', api_key: '', priority: providers.length + 1 });
    setShowModal(true);
  };

  const openEditModal = (index: number) => {
    setEditingIndex(index);
    setFormData({ ...providers[index] });
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setEditingIndex(null);
  };

  const handleSave = () => {
    if (!formData.name || !formData.base_url || !formData.model) {
      alert('Please fill in all required fields');
      return;
    }

    const updatedProviders = [...providers];
    if (editingIndex !== null) {
      updatedProviders[editingIndex] = formData;
    } else {
      updatedProviders.push(formData);
    }
    onProvidersChange(updatedProviders);
    closeModal();
  };

  const handleDelete = (index: number) => {
    if (window.confirm('Are you sure you want to delete this backup provider?')) {
      const updatedProviders = providers.filter((_, i) => i !== index);
      onProvidersChange(updatedProviders);
    }
  };

  const moveUp = (index: number) => {
    if (index === 0) return;
    const updatedProviders = [...providers];
    [updatedProviders[index - 1], updatedProviders[index]] = [updatedProviders[index], updatedProviders[index - 1]];
    onProvidersChange(updatedProviders);
  };

  const moveDown = (index: number) => {
    if (index === providers.length - 1) return;
    const updatedProviders = [...providers];
    [updatedProviders[index], updatedProviders[index + 1]] = [updatedProviders[index + 1], updatedProviders[index]];
    onProvidersChange(updatedProviders);
  };

  return (
    <div style={{
      backgroundColor: 'white',
      padding: '25px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      marginBottom: '20px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3 style={{ 
          margin: '0', 
          color: '#333', 
          fontSize: '18px',
          borderBottom: '2px solid #007bff',
          paddingBottom: '10px',
          flex: 1
        }}>
          Backup Providers
        </h3>
        <button
          onClick={openAddModal}
          style={{
            padding: '8px 16px',
            backgroundColor: '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold'
          }}
        >
          + Add Provider
        </button>
      </div>

      <p style={{ fontSize: '12px', color: '#6c757d', margin: '0 0 15px 0' }}>
        Configure fallback API providers in priority order
      </p>

      {providers.length === 0 ? (
        <div style={{
          padding: '30px',
          textAlign: 'center',
          backgroundColor: '#f8f9fa',
          borderRadius: '5px',
          border: '1px dashed #ddd',
          color: '#6c757d'
        }}>
          No backup providers configured. Add one to enable automatic failover.
        </div>
      ) : (
        <div>
          {providers.map((provider, index) => (
            <div
              key={index}
              style={{
                border: '1px solid #ddd',
                borderRadius: '5px',
                padding: '15px',
                marginBottom: '10px',
                backgroundColor: 'white'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <h4 style={{ margin: '0 0 8px 0', color: '#333', fontSize: '16px' }}>
                    {provider.name}
                  </h4>
                  <p style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#6c757d' }}>
                    <strong>URL:</strong> {provider.base_url}
                  </p>
                  <p style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#6c757d' }}>
                    <strong>Model:</strong> {provider.model}
                  </p>
                  <p style={{ margin: '0', fontSize: '13px', color: '#6c757d' }}>
                    <strong>Priority:</strong> {provider.priority}
                  </p>
                </div>
                
                <div style={{ display: 'flex', gap: '5px', flexDirection: 'column' }}>
                  <div style={{ display: 'flex', gap: '5px' }}>
                    <button
                      onClick={() => moveUp(index)}
                      disabled={index === 0}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: index === 0 ? 'not-allowed' : 'pointer',
                        fontSize: '12px',
                        opacity: index === 0 ? 0.5 : 1
                      }}
                    >
                      ↑
                    </button>
                    <button
                      onClick={() => moveDown(index)}
                      disabled={index === providers.length - 1}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: index === providers.length - 1 ? 'not-allowed' : 'pointer',
                        fontSize: '12px',
                        opacity: index === providers.length - 1 ? 0.5 : 1
                      }}
                    >
                      ↓
                    </button>
                  </div>
                  <button
                    onClick={() => openEditModal(index)}
                    style={{
                      padding: '4px 8px',
                      backgroundColor: '#ffc107',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(index)}
                    style={{
                      padding: '4px 8px',
                      backgroundColor: '#dc3545',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {showModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '25px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            maxWidth: '500px',
            width: '90%'
          }}>
            <h3 style={{ margin: '0 0 20px 0', color: '#333' }}>
              {editingIndex !== null ? 'Edit Backup Provider' : 'Add Backup Provider'}
            </h3>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', fontWeight: '500' }}>
                Provider Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="e.g., OpenAI Backup"
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', fontWeight: '500' }}>
                Base URL *
              </label>
              <input
                type="text"
                value={formData.base_url}
                onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                placeholder="https://api.openai.com/v1"
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', fontWeight: '500' }}>
                Model *
              </label>
              <input
                type="text"
                value={formData.model}
                onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                placeholder="gpt-4"
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', fontWeight: '500' }}>
                API Key (optional)
              </label>
              <input
                type="password"
                value={formData.api_key || ''}
                onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                placeholder="sk-..."
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', fontWeight: '500' }}>
                Priority
              </label>
              <input
                type="number"
                value={formData.priority}
                onChange={(e) => setFormData({ ...formData, priority: parseInt(e.target.value) || 1 })}
                min="1"
                style={{
                  width: '100%',
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={closeModal}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BackupProvidersSection;

