import React from 'react';

const ProtonBridgeSetup: React.FC = () => <div className="mcp-bridge-setup">
  <h4>Set up Proton Mail Bridge</h4>
  <ol>
    <li><a href="https://proton.me/mail/bridge" target="_blank" rel="noreferrer">Install Proton Mail Bridge</a> on the machine running your local MCP server. A paid plan that includes Proton Mail is required.</li>
    <li>Open Bridge and sign in to Proton there. Keep Bridge running.</li>
    <li>Open the account’s Bridge mailbox settings and copy the IMAP/SMTP host, ports, username, and generated password into your chosen MCP server’s configuration. Use the Bridge password, which is separate from your Proton account password.</li>
    <li>Enter your installed IMAP/SMTP MCP server’s command and arguments below. Environment variable names depend on that server; use its documented names.</li>
    <li>Save the server, then select Test to check tool discovery. Enable it when you are ready to use its tools.</li>
  </ol>
  <p>If Geist runs in Docker, localhost refers to the container. The MCP server must be able to reach Bridge on the host; running the MCP server alongside Bridge avoids this mismatch. Keep Bridge bound to the local machine.</p>
  <p>Geist does not install Bridge or an MCP server, and never needs your Proton account password.</p>
</div>;

export default ProtonBridgeSetup;
