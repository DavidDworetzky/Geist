
from twilio.rest import Client

from adapters.base_adapter import BaseAdapter


class SMSAdapter(BaseAdapter):
    '''
    SMS Adapter sends a text via twilio to a target
    '''
    def __init__(self, twilio_key:str, twilio_sid:str, twilio_source:str, **kwargs):
        self.api_key = twilio_key
        self.account_sid = twilio_sid
        self.source_number = twilio_source
        self.client = Client(twilio_sid, twilio_key)


    def enumerate_actions(self) -> list[str]:
        return ["send_text"]

    def send_text(self, message: str, number: str):
        created = self.client.messages.create(
            body=message,
            from_=self.source_number,
            to=number
        )
        return created.sid
