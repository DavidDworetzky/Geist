import requests
from agents.base_agent import BaseAgent
from agents.agent_context import AgentContext
import subprocess
import os
import signal
import psutil

WORLD_TICK_PROMPT = f"""You are a deep and thorough thinker. 
Given what you know about the world today, and the main task that you need to complete, consider if there are any additional important facts that you should add to the list of your knowledge. 
Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler truths."""

TASK_TICK_PROMPT = f"You are a driven and focused individual. Given the main task that you wish to complete, and current working subtasks, create a specific list of actionable tasks that will complete your problem. Delimit these as plain english separated by the | character"

EXECUTION_TICK_PROMPT = "You are given a list of tasks and list of function calls that you can make. Given the state of the world, formulate a function call that will help you complete your task."

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

    def _aggregated_context(self, world_context : bool, task_context : bool, execution_context: bool):
        #get aggregated context for world, task and execution context if requested
        context_string = ""
        if world_context:
            context_string += "WORLD_CONTEXT:" + self.agent_context.world_context.join("\n")
        if task_context:
            context_string += "TASK_CONTEXT:" + self.agent_context.task_context.join("\n")
        if execution_context:
            context_string += "EXECUTION_CONTEXT:" + self.agent_context.execution_context.join("\n")


    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
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
        
    def initialize(self):
        if self.as_subprocess:
            # Create a subprocess that runs one tick every second
            process = subprocess.Popen(['python3', '-u', 'tick.py'], stdout=subprocess.PIPE)

            # Set the subprocess ID in our agent context
            self.agent_context.subprocess_id
        else:
            return None

    def terminate_subprocess(self):
        # Retrieve subprocess ID from agent context
        subprocess_id = self.agent_context.subprocess_id

        # Check if the subprocess ID is set
        if subprocess_id:
            # Send a terminate signal to the subprocess
            os.kill(subprocess_id, signal.SIGTERM)
            self.agent_context.subprocess_id = None
        else:
            raise Exception("No subprocess ID set in agent context.")
        
    def _pop_and_add_execution_tasks(self):
            if self._agent_context.task_context:
                task_to_execute = self._agent_context.task_context.pop(0)
                prompt = f"executing task: {task_to_execute}" + self._aggregated_context(world_context=True, task_context=True, execution_context=True)
                execution_result = self.complete_text(prompt=TASK_TICK_PROMPT + prompt)
                tasks_to_execute = execution_result.split('|')
                self._agent_context.execution_context.append(tasks_to_execute)
            else:
                raise Exception("No tasks available in task context for execution.")
            
    def _clear_execution_tasks(self):
        results = []
        for task in self._agent_context.execution_context:
            context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
            result = self.complete_text(prompt=f"task: {task}" + EXECUTION_TICK_PROMPT + context_string)
            results.append(result)

        self._agent_context.execution_context = []
        return results
        

    def tick(self):
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


    def tick_world(self):
        '''
        Advances the state of the world context, reasoning about the world state.
        '''
        #aggregate list of strings from world_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=False)
        result = self.complete_text(prompt=WORLD_TICK_PROMPT + context_string)
        #split result and replace world context
        split_result = result.split("\n")
        self.agent_context.world_context = split_result
        return split_result

    def tick_tasks(self):
        '''
        Advances the state of the task context, reasoning about how to complete the primary task.
        '''
        #aggregate list of strings from task_context
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(prompt=TASK_TICK_PROMPT + context_string)
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
        #split result and replace execution context
        split_result = result.split("\n")
        self.agent_context.execution_context = split_result
        return split_result
