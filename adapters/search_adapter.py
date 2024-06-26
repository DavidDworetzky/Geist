from abc import ABC, abstractmethod
from adapters.base_adapter import BaseAdapter
from typing import List
from datetime import datetime
import requests 

class SearchAdapter(BaseAdapter):
    '''
    Search Adapter alllows us to search and get web page contents.
    '''
    def __init__(self, base_url: str = None, **kwargs):
        if base_url is None:
            base_url = "https://www.google.com"
        self.base_url = base_url
        super().__init__(**kwargs)
    
    def enumerate_actions(self) -> List[str]:
        return ["search", "get"]
    
    def search(self, search_term: str) -> str:

        # Construct the search URL
        search_url = f"{self.base_url}/search?q={search_term}"

        # Perform the HTTP GET request
        response = requests.get(search_url)

        # Return the response text or handle errors appropriately
        if response.status_code == 200:
            return response.text
        else:
            return "Failed to retrieve search results"
    
    def get(self, url: str) -> str:
        # Perform the HTTP GET request
        response = requests.get(url)

        # Return the response text or handle errors appropriately
        if response.status_code == 200:
            return response.text
        else:
            return "Failed to retrieve content"