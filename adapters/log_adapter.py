from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter
from typing import List
import datetime

class LogAdapter(BaseAdapter):
    '''
    Log Adapter is an adapter to let agents log to our local filesystem. 
    '''
    def __init__(self, filename:str = None):
        if filename is None:
            date_str = datetime.now().strftime("%m%d%Y")  # Generates the current date in MMDDYYYY format
            filename = f"geist_log_{date_str}.txt"  # Default filename with current date     
        self.filename = filename       
            

    def enumerate_actions(self) -> List[str]:
        return ["log"]
    
    def log(self, output: str):
        with open(self.filename, "a") as file:
            file.write(output + "\n")

