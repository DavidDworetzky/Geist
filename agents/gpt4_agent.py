import requests
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
import subprocess
import os
import signal
import psutil
import json
import logging
from utils.logging import log_function_call

WORLD_TICK_PROMPT = f"""You are a world class executive. Your plans are plans are direct, and detailed only if necessary. 
Given what you know about the world today, and the main task that you need to complete, consider if there are any additional facts that you should add to the list of things you consider. 
Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler statements."""

TASK_TICK_PROMPT = f"You are a focused individual. Given the main task that you wish to complete, and current working subtasks, create a specific list of actionable tasks that will complete your problem. Delimit these as plain english separated by the | character. Do not use function calls yet - only plain english."

FUNCTION_CALL_JSON = """
{
    "class" : "class_name",
    "function": "function_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}
"""

EXECUTION_TICK_PROMPT = f"You are given a list of tasks and list of function calls that you can make. Given the state of the world, and classes available to you - formulate a function call that will help you complete your task. You should formulate the function call as {FUNCTION_CALL_JSON}. Only call functions that are listed in our adapter list."


class GPT4Agent(BaseAgent):
    def __init__(self, api_key, agent_context, as_subprocess=False):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # call super constructor
        super().__init__(agent_context, as_subprocess)

    def phase_out(self):
        self._agent_context._save()
        self.terminate_subprocess()

    def phase_in(self):
        self.initialize()

    def _is_valid_function_json(self, function_json:str):
        try:
            function_json = function_json.replace('\n', '')
            # Attempt to parse the JSON string
            parsed_json = json.loads(function_json)
            # Check if the required keys are present
            required_keys = ["function", "parameters", "class"]
            if all(key in parsed_json for key in required_keys):
                # Check if parameters have the correct structure
                if isinstance(parsed_json["parameters"], dict) :
                    return True
            return False
        except json.JSONDecodeError:
            # Return False if JSON is invalid
            return False        
        

    def _take_json_and_call_function(self, function_json:str):
        '''
        takes a json definition of a function call and uses it to call one of our function adapters.
        '''
        if not self._is_valid_function_json(function_json):
            raise Exception(f"invalid function call json: {function_json}")
    
        json_data = json.loads(function_json)
    
        # now, find our adapter class and call the relevant function on it.
        class_name = json_data["class"]
        adapter_class = next((wrapper for wrapper in self._agent_context.initialized_classes if wrapper.name == class_name), None)

        if not adapter_class:
            raise Exception(f"No adapter class matching{class_name}")
    
        adapter_class = adapter_class.instance
        # now, call the relevant function on the adapter_class through reflection
        function_to_call = getattr(adapter_class, json_data["function"])
        parameters = json_data["parameters"]
    
        # Execute the function with the provided parameters
        result = function_to_call(**parameters)
    
        return result
        

    def _aggregated_context(self, world_context : bool, task_context : bool, execution_context: bool):
        #get aggregated context for world, task and execution context if requested
        context_string = ""
        if world_context:
            context_string += "WORLD_CONTEXT:" + "\n".join(self._agent_context.world_context)
        if task_context:
            context_string += "TASK_CONTEXT:" + "\n".join(self._agent_context.task_context)
        if execution_context:
            context_string += "EXECUTION_CONTEXT:" + "\n".join(self._agent_context.execution_context)
        return context_string

    def complete_text(self, prompt:str, max_tokens:int = None, n:int = None, temperature = None, top_p: int = None, frequency_penalty = None, presence_penalty = None, stop:str = None, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
        #set defaults for agent settings based off of settings values. If undefined,\
        max_tokens = self._agent_context.settings.max_tokens if self._agent_context.settings.max_tokens and not max_tokens else 16
        n = self._agent_context.settings.n if self._agent_context.settings.n and not n else 1
        temperature = self._agent_context.settings.temperature if self._agent_context.settings.temperature and not temperature else 1.0
        top_p = self._agent_context.settings.top_p if self._agent_context.settings.top_p and not top_p else 1
        frequency_penalty = self._agent_context.settings.frequency_penalty if self._agent_context.settings.frequency_penalty and not frequency_penalty else 0
        presence_penalty = self._agent_context.settings.presence_penalty if self._agent_context.settings.presence_penalty and not presence_penalty else 0
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4",
            "max_tokens": max_tokens,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        if stop is not None:
            payload["stop"] = stop

        response = requests.post(self.base_url, json=payload, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
    def initialize(self, task:str = None):
        #push task onto our stack for this agent.
        if task:
            self._agent_context.task_context.append(task)
        if self.as_subprocess:
            logging.info("Initializing agent with subprocess.")
            # Create a subprocess that runs one tick every second
            process = subprocess.Popen(['python3', '-u', 'tick.py'], stdout=subprocess.PIPE)

            # Set the subprocess ID in our agent context
            self._agent_context.subprocess_id = process.pid
        else:
            logging.info("Initializing agent without subprocess.")
            self._agent_context.subprocess_id = None
            return None

    def terminate_subprocess(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self._agent_context.subprocess_id

        # Check if the subprocess ID is set
        if subprocess_id:
            # Send a terminate signal to the subprocess
            os.kill(subprocess_id, signal.SIGTERM)
            self._agent_context.subprocess_id = None
        else:
            raise Exception("No subprocess ID set in agent context.")
    
    @log_function_call
    def _pop_and_add_execution_tasks(self):
            if self._agent_context.task_context:
                task_to_execute = self._agent_context.task_context.pop(0)
                prompt = f"executing task: {task_to_execute}" + self._aggregated_context(world_context=True, task_context=True, execution_context=True)
                execution_result = self.complete_text(prompt=TASK_TICK_PROMPT + prompt)
                execution_result = self._transform_completions(execution_result)
                tasks_to_execute = [task.strip() for result in execution_result for task in result.split('|')]
                self._agent_context.execution_context = tasks_to_execute
                return True
            else:
                raise Exception("No tasks available in task context for execution.")
            
    @log_function_call
    def _clear_execution_tasks(self):
        results = []
        for task in self._agent_context.execution_context:
            context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
            result = self.complete_text(prompt=f"task: {task}" + EXECUTION_TICK_PROMPT + context_string)
            result = self._transform_completions(result)
            #get first result as function call
            result = result[0]
            retries = 0

            while not self._is_valid_function_json(result) and retries < 3:
                retries += 1
                result = self.complete_text(prompt=f"task: {task}" + EXECUTION_TICK_PROMPT + context_string)
                result = self._transform_completions(result)
                result = result[0]

            if not self._is_valid_function_json(result):
                logging.error(f'Invaild result for function call is: {result}')
                raise Exception("Exceeded retries for valid function call JSON.")
            
            output = self._take_json_and_call_function(result)
            results.append(output)

        self._agent_context.execution_context = []
        return results
        

    @log_function_call
    def tick(self):
        logging.info("Agent Tick.")
        # Make one inference call to GPT-4 to advance world state reasoning, tasks and then execute.
        # reason about the world, then.
        # pop elements of agent context tasks for execution
        self.tick_world()
        self._pop_and_add_execution_tasks()
        self._clear_execution_tasks()

    def is_subprocess_running(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self.agent_context.subprocess_id
        # check if subprocess is running and return value (by using psutil)
        try:
            p = psutil.Process(subprocess_id)
            return p.is_running()
        except psutil.NoSuchProcess:
            return False

    def _transform_completions(self, completion):
        try:
            choices_list = completion['choices']
            transformed_content = list(map(lambda x: x['message']['content'], choices_list))
            return transformed_content
        except Exception as e:
            logging.error(f"contents of completion, failed to destructure: {completion}, exception {e}")
            raise Exception(f"completion failed to destructure: {completion}. Format interop failure. Is your LLM protocol returning the correct format?")

    @log_function_call
    def tick_world(self):
        '''
        Advances the state of the world context, reasoning about the world state.
        '''
        #aggregate list of strings from world_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=False)
        input = WORLD_TICK_PROMPT + context_string
        result = self.complete_text(prompt=input)
        #split result and replace world context
        split_result = self._transform_completions(result)
        self._agent_context.world_context = split_result
        return split_result

    def tick_tasks(self):
        '''
        Advances the state of the task context, reasoning about how to complete the primary task.
        '''
        #aggregate list of strings from task_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=TASK_TICK_PROMPT + context_string)
        result = self._transform_completions(result)
        #split result and replace task context
        split_result = result.split("\n")
        self.agent_context.task_context = split_result
        return split_result
    
    
    def tick_execution(self):
        '''
        Advances the state of the execution context, reasoning about how to execute the primary task.
        '''
        #aggregate list of strings from execution_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=EXECUTION_TICK_PROMPT + context_string)
        result = self._transform_completions(result)
        #split result and replace execution context
        split_result = result.split("\n")
        self._agent_context.execution_context = split_result
        return split_result
