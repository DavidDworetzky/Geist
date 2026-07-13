from datetime import datetime

from adapters.base_adapter import BaseAdapter


class LogAdapter(BaseAdapter):
    '''
    Log Adapter is an adapter to let agents log to our local filesystem.
    '''
    def __init__(self, filename: str | None = None, **kwargs):
        if filename is None:
            date_str = datetime.now().strftime("%m%d%Y")  # Generates the current date in MMDDYYYY format
            filename = f"geist_log_{date_str}.txt"  # Default filename with current date
        self.filename = filename


    def enumerate_actions(self) -> list[str]:
        return ["log", "read_log"]

    def log(self, output: str):
        with open(self.filename, "a") as file:
            file.write(output + "\n")

    def read_log(self) -> list[str]:
        with open(self.filename) as file:
            return file.readlines()
