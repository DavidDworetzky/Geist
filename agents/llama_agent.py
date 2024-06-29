from agents.base_agent import BaseAgent
from llama_cpp import Llama



'''
llama class model agents.
'''
WORLD_TICK_PROMPT = f"""You are a deep and thorough thinker. 
Given what you know about the world today, and the main task that you need to complete, consider if there are any additional important facts that you should add to the list of your knowledge. 
Do not add anything that doesn't need to be added, consolidate anything that is worth consolidating with simpler truths."""

TASK_TICK_PROMPT = f"You are a driven and focused individual. Given the main task that you wish to complete, and current working subtasks, add any additional tasks that will help you complete your main task."

EXECUTION_TICK_PROMPT = "You are given a list of tasks and list of function calls that you can make. Given the state of the world, formulate a function call that will help you complete your task."

class LlamaAgent(BaseAgent):
    def __init__(self, model_path, agent_context):
        self.model_path = model_path
        self.llm = Llama(model_path)
        # call super constructor
        super().__init__(agent_context)

    def _aggregated_context(self, world_context : bool, task_context : bool, execution_context: bool):
        #get aggregated context for world, task and execution context if requested
        context_string = ""
        if world_context:
            context_string += "WORLD_CONTEXT:" + self.agent_context.world_context.join("\n")
        if task_context:
            context_string += "TASK_CONTEXT:" + self.agent_context.task_context.join("\n")
        if execution_context:
            context_string += "EXECUTION_CONTEXT:" + self.agent_context.execution_context.join("\n")

    def initialize(self, task_prompt):
        '''
        Initialize the agent with a given task prompt.
        '''
        self.agent_context.task_context.append(task_prompt)

    def tick(self):
        '''
        Perform a tick operation, advancing the agent's understanding and actions.
        '''
        self.tick_world()
        self.tick_tasks()
        self.tick_execution()

    def tick_world(self):
        '''
        Advances the state of the world context.
        '''
        context_string = self._aggregated_context(world_context=True, task_context=False, execution_context=False)
        result = self.complete_text(WORLD_TICK_PROMPT + context_string)
        self.agent_context.world_context = result.split("\n")

    def tick_tasks(self):
        '''
        Advances the state of the task context.
        '''
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=False)
        result = self.complete_text(TASK_TICK_PROMPT + context_string)
        self.agent_context.task_context = result.split("\n")

    def tick_execution(self):
        '''
        Advances the state of the execution context.
        '''
        context_string = self._aggregated_context(world_context=True, task_context=True, execution_context=True)
        result = self.complete_text(EXECUTION_TICK_PROMPT + context_string)
        self.agent_context.execution_context = result.split("\n")

    def push_task(self, task_prompt):
        '''
        Add a new task to the agent's task context.
        '''
        self.agent_context.task_context.append(task_prompt)


    def complete_text(self, prompt, max_tokens=16, n=1, stop=None, temperature=1.0, top_p=1, frequency_penalty=0, presence_penalty=0, echo=False, best_of=None, prompt_tokens=None, response_format="text"):
        response = self.llm.prompt(prompt, max_tokens = max_tokens)
        return response