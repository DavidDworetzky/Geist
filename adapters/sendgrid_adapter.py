from adapters.base_adapter import BaseAdapter
from typing import List, Dict, Any, Optional
import re
import requests
import json


class SendGridAdapter(BaseAdapter):
    '''
    SendGrid Email Adapter for sending emails with markdown templates and token replacement
    '''
    
    def __init__(self, sendgrid_api_key: str, from_email: str, from_name: str = None, **kwargs):
        self.api_key = sendgrid_api_key
        self.from_email = from_email
        self.from_name = from_name or "Geist"
        self.base_url = "https://api.sendgrid.com/v3/mail/send"
        
    def enumerate_actions(self) -> List[str]:
        return ["send_email", "send_template_email"]
    
    def _replace_tokens(self, content: str, tokens: Dict[str, Any]) -> str:
        """
        Replace {{token}} patterns in content with values from tokens dict
        """
        if not tokens:
            return content
            
        def replace_token(match):
            token_name = match.group(1).strip()
            return str(tokens.get(token_name, match.group(0)))
        
        return re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_token, content)
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """
        Convert basic markdown to HTML
        """
        html = markdown_content
        
        # Headers
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        
        # Line breaks
        html = re.sub(r'\n\n', '<br><br>', html)
        html = re.sub(r'\n', '<br>', html)
        
        return html
    
    def _send_email_request(self, payload: Dict[str, Any]) -> str:
        """
        Send email via SendGrid API
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code == 202:
                return "Email sent successfully"
            else:
                return f"Error sending email: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error sending email: {str(e)}"
    
    def send_email(self, to_email: str, subject: str, content: str, to_name: str = None) -> str:
        """
        Send a plain text or HTML email
        """
        # Check if content looks like HTML
        is_html = '<' in content and '>' in content
        
        payload = {
            "personalizations": [
                {
                    "to": [{"email": to_email, "name": to_name or to_email}],
                    "subject": subject
                }
            ],
            "from": {
                "email": self.from_email,
                "name": self.from_name
            },
            "content": [
                {
                    "type": "text/html" if is_html else "text/plain",
                    "value": content
                }
            ]
        }
        
        return self._send_email_request(payload)
    
    def send_template_email(self, to_email: str, subject: str, template: str, 
                          tokens: Dict[str, Any] = None, to_name: str = None, 
                          is_markdown: bool = True) -> str:
        """
        Send email with template processing and token replacement
        
        Args:
            to_email: Recipient email address
            subject: Email subject (can contain tokens)
            template: Email template content (markdown or HTML)
            tokens: Dictionary of tokens to replace in template
            to_name: Recipient name (optional)
            is_markdown: Whether to convert markdown to HTML (default: True)
        """
        # Replace tokens in subject
        processed_subject = self._replace_tokens(subject, tokens or {})
        
        # Replace tokens in template
        processed_content = self._replace_tokens(template, tokens or {})
        
        # Convert markdown to HTML if needed
        if is_markdown:
            processed_content = self._markdown_to_html(processed_content)
        
        # Send email
        return self.send_email(to_email, processed_subject, processed_content, to_name)