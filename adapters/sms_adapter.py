from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter
from typing import List
from datetime import datetime
from twilio.rest import Client

class SMSAdapter(BaseAdapter):
    '''
    SMS Adapter sends a text via twilio to a target
    '''
    def __init__(self, api_key:str, account_sid:str, source_number:str, **kwargs):
        self.api_key = api_key
        self.account_sid = account_sid
        self.source_number = source_number
        self.client = Client(account_sid, api_key)

    def enumerate_actions(self) -> List[str]:
        return ["send_text"]
    
    def send_text(self, message: str, number: str):
        message = self.client.messages.create(
            body=message,
            from_=self.source_number,  
            to=number
        )
        return message.sid
