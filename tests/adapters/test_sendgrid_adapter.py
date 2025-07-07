import pytest
from unittest.mock import Mock, patch
from adapters.sendgrid_adapter import SendGridAdapter


class TestSendGridAdapter:
    
    def setup_method(self):
        self.adapter = SendGridAdapter(
            sendgrid_api_key="test_key",
            from_email="test@example.com",
            from_name="Test Sender"
        )
    
    def test_enumerate_actions(self):
        actions = self.adapter.enumerate_actions()
        assert "send_email" in actions
        assert "send_template_email" in actions
    
    def test_replace_tokens_basic(self):
        content = "Hello {{name}}, welcome to {{company}}!"
        tokens = {"name": "John", "company": "Geist"}
        result = self.adapter._replace_tokens(content, tokens)
        assert result == "Hello John, welcome to Geist!"
    
    def test_replace_tokens_with_spaces(self):
        content = "Hello {{ name }}, your balance is {{  balance  }}."
        tokens = {"name": "Alice", "balance": "$100"}
        result = self.adapter._replace_tokens(content, tokens)
        assert result == "Hello Alice, your balance is $100."
    
    def test_replace_tokens_missing_token(self):
        content = "Hello {{name}}, your {{missing}} is here."
        tokens = {"name": "Bob"}
        result = self.adapter._replace_tokens(content, tokens)
        assert result == "Hello Bob, your {{missing}} is here."
    
    def test_replace_tokens_empty_dict(self):
        content = "Hello {{name}}"
        result = self.adapter._replace_tokens(content, {})
        assert result == "Hello {{name}}"
    
    def test_replace_tokens_none_dict(self):
        content = "Hello {{name}}"
        result = self.adapter._replace_tokens(content, None)
        assert result == "Hello {{name}}"
    
    def test_markdown_to_html_headers(self):
        markdown = "# Header 1\n## Header 2\n### Header 3"
        result = self.adapter._markdown_to_html(markdown)
        assert "<h1>Header 1</h1>" in result
        assert "<h2>Header 2</h2>" in result
        assert "<h3>Header 3</h3>" in result
    
    def test_markdown_to_html_bold_italic(self):
        markdown = "This is **bold** and *italic* text."
        result = self.adapter._markdown_to_html(markdown)
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
    
    def test_markdown_to_html_links(self):
        markdown = "Visit [Google](https://google.com) for search."
        result = self.adapter._markdown_to_html(markdown)
        assert '<a href="https://google.com">Google</a>' in result
    
    def test_markdown_to_html_line_breaks(self):
        markdown = "Line 1\n\nLine 2\nLine 3"
        result = self.adapter._markdown_to_html(markdown)
        assert "<br><br>" in result
        assert "<br>" in result
    
    @patch('adapters.sendgrid_adapter.requests.post')
    def test_send_email_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 202
        mock_post.return_value = mock_response
        
        result = self.adapter.send_email(
            to_email="test@example.com",
            subject="Test Subject",
            content="Test content"
        )
        
        assert result == "Email sent successfully"
        mock_post.assert_called_once()
    
    @patch('adapters.sendgrid_adapter.requests.post')
    def test_send_email_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        result = self.adapter.send_email(
            to_email="test@example.com",
            subject="Test Subject",
            content="Test content"
        )
        
        assert "Error sending email: 400" in result
    
    @patch('adapters.sendgrid_adapter.requests.post')
    def test_send_template_email_with_tokens(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 202
        mock_post.return_value = mock_response
        
        template = "# Hello {{name}}\n\nWelcome to {{company}}!"
        tokens = {"name": "John", "company": "Geist"}
        
        result = self.adapter.send_template_email(
            to_email="test@example.com",
            subject="Welcome {{name}}!",
            template=template,
            tokens=tokens
        )
        
        assert result == "Email sent successfully"
        
        # Check that the request was made with processed content
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        # Check subject was processed
        assert payload['personalizations'][0]['subject'] == "Welcome John!"
        
        # Check content was processed and converted to HTML
        content_value = payload['content'][0]['value']
        assert "<h1>Hello John</h1>" in content_value
        assert "Welcome to Geist!" in content_value
    
    def test_send_template_email_plain_text(self):
        with patch.object(self.adapter, 'send_email') as mock_send:
            mock_send.return_value = "Email sent successfully"
            
            result = self.adapter.send_template_email(
                to_email="test@example.com",
                subject="Test",
                template="Hello {{name}}",
                tokens={"name": "John"},
                is_markdown=False
            )
            
            mock_send.assert_called_once_with(
                "test@example.com",
                "Test",
                "Hello John",
                None
            )
    
    def test_html_detection_in_send_email(self):
        # Test that HTML content is detected correctly
        html_content = "<h1>Test</h1><p>This is HTML</p>"
        
        with patch.object(self.adapter, '_send_email_request') as mock_send:
            mock_send.return_value = "Email sent successfully"
            
            self.adapter.send_email(
                to_email="test@example.com",
                subject="Test",
                content=html_content
            )
            
            # Check that HTML content type was used
            call_args = mock_send.call_args[0][0]
            assert call_args['content'][0]['type'] == 'text/html'
    
    def test_plain_text_detection_in_send_email(self):
        # Test that plain text content is detected correctly
        plain_content = "This is plain text without HTML tags"
        
        with patch.object(self.adapter, '_send_email_request') as mock_send:
            mock_send.return_value = "Email sent successfully"
            
            self.adapter.send_email(
                to_email="test@example.com",
                subject="Test",
                content=plain_content
            )
            
            # Check that plain text content type was used
            call_args = mock_send.call_args[0][0]
            assert call_args['content'][0]['type'] == 'text/plain'