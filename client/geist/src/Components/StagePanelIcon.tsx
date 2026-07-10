export type StagePanelIconName = 'plus' | 'maximize' | 'search' | 'folder' | 'more' | 'close' | 'workflow' | 'edit' | 'check' | 'save' | 'play';

export default function StagePanelIcon({ name }: { name: StagePanelIconName }): JSX.Element {
  if (name === 'plus') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6V5Z" />
      </svg>
    );
  }

  if (name === 'maximize') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 9V4h5v2H6v3H4Zm11-5h5v5h-2V6h-3V4ZM4 15h2v3h3v2H4v-5Zm14 0h2v5h-5v-2h3v-3Z" />
      </svg>
    );
  }
  if (name === 'search') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M10.5 4a6.5 6.5 0 0 1 5.17 10.44l4.45 4.44-1.42 1.42-4.44-4.45A6.5 6.5 0 1 1 10.5 4Zm0 2a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Z" />
      </svg>
    );
  }

  if (name === 'folder') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 6.75C3 5.78 3.78 5 4.75 5h5.1c.46 0 .9.18 1.24.51L12.58 7h6.67c.97 0 1.75.78 1.75 1.75v8.5c0 .97-.78 1.75-1.75 1.75H4.75C3.78 19 3 18.22 3 17.25V6.75Z" />
      </svg>
    );
  }

  if (name === 'more') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M6.5 10a2 2 0 1 1 0 4 2 2 0 0 1 0-4Zm5.5 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4Zm5.5 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z" />
      </svg>
    );
  }

  if (name === 'close') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6.7 5.3 5.3 5.29 5.3-5.3 1.4 1.42-5.29 5.3 5.3 5.29-1.42 1.4-5.3-5.29-5.29 5.3-1.4-1.42 5.29-5.3-5.3-5.29 1.42-1.4Z" />
      </svg>
    );
  }

  if (name === 'edit') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m15.7 4.3 4 4L9 19H5v-4L15.7 4.3Zm0 2.83L7 15.83V17h1.17l8.7-8.7-1.17-1.17Z" />
      </svg>
    );
  }

  if (name === 'check') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m9.2 16.2-4.4-4.4 1.4-1.4 3 3 8.6-8.6 1.4 1.4-10 10Z" />
      </svg>
    );
  }
  if (name === 'save') {
    return (
      <svg viewBox='0 0 24 24' aria-hidden='true'>
        <path d='M5 3h12.2L21 6.8V21H3V3h2Zm0 2v14h14V7.62L16.38 5H16v5H7V5H5Zm4 0v3h5V5H9Zm-1 8h8v5H8v-5Z' />
      </svg>
    );
  }

  if (name === 'play') {
    return (
      <svg viewBox='0 0 24 24' aria-hidden='true'>
        <path d='M7 4.55v14.9L19.42 12 7 4.55ZM9 8.08 15.58 12 9 15.92V8.08Z' />
      </svg>
    );
  }

  if (name === 'workflow') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 7.75A2.75 2.75 0 1 1 7 2.25a2.75 2.75 0 0 1 0 5.5Zm10 14A2.75 2.75 0 1 1 17 16.25a2.75 2.75 0 0 1 0 5.5ZM7 21.75A2.75 2.75 0 1 1 7 16.25a2.75 2.75 0 0 1 0 5.5ZM9.5 5h5.25A3.25 3.25 0 0 1 18 8.25v2.5A3.25 3.25 0 0 1 14.75 14H9.5v-2h5.25c.69 0 1.25-.56 1.25-1.25v-2.5C16 7.56 15.44 7 14.75 7H9.5V5Z" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5 20.5 12 12 20.5 3.5 12 12 3.5Zm0 2.83L6.33 12 12 17.67 17.67 12 12 6.33Z" />
    </svg>
  );
}
