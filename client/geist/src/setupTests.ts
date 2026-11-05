import { TextEncoder, TextDecoder } from 'util';
// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// jsdom does not implement the native dialog lifecycle.
HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };

// React Router 7 uses the encoding APIs absent from Jest 27's jsdom.
Object.assign(globalThis, { TextEncoder, TextDecoder });
