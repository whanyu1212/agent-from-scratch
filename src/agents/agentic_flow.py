import sys

sys.path.insert(0, "..")
from colorama import init, Fore, Style
from typing import Any
from loguru import logger
from src.prompts.prompt import agent_system_prompt_template
from src.models.gemini_models import GeminiModel
from src.tools.basic_calculator import basic_calculator
from src.tools.date_parser import parse_datetime
from src.toolbox.toolbox import ToolBox

init()

# Fore.RED
# Fore.GREEN
# Fore.YELLOW
# Fore.BLUE
# Fore.CYAN
# Fore.MAGENTA


class Agent:
    def __init__(self, tools: list, model_name: str):
        """Initializes an Agent instance.

        Args:
            tools (list): a list of tool functions that the agent can use.
            model_name (str): the model name available in the Gemini API.
        """
        self.tools = tools
        self.model_name = model_name

    def format_tool_descriptions(self) -> str:
        """Prepares the tool descriptions for the system prompt.

        Returns:
            str: The tool names and descriptions as a string.
        """
        toolbox = ToolBox()
        toolbox.register_functions(self.tools)
        tool_descriptions = toolbox.get_registered_functions_as_string()
        return tool_descriptions

    def plan_action(self, prompt: str) -> dict:
        """
        Analyzes the user prompt and plans the appropriate tool execution.

        Parameters:
        prompt (str): The user query to analyze.

        Returns:
        dict: Contains selected tool and arguments for execution.
        """
        tool_descriptions = self.format_tool_descriptions()
        agent_system_prompt = agent_system_prompt_template.format(
            tool_descriptions=tool_descriptions
        )

        logger.info(
            f"{Fore.CYAN}System prompt:\n {agent_system_prompt} {Style.RESET_ALL}"
        )

        model_instance = GeminiModel(
            model_name=self.model_name, system_prompt=agent_system_prompt, temperature=0
        )

        # Generate and return the execution plan
        action_plan = model_instance.generate_text(prompt)
        return action_plan

    def execute_planned_action(self, prompt: str) -> Any:
        """
        Executes the planned action by routing to the appropriate tool.

        Parameters:
        prompt (str): The user query to execute.

        Returns:
        Any: The response from executing the selected tool.
        """
        agent_response_dict = self.plan_action(prompt)
        print(f"{Fore.BLUE}Tool Routing: {agent_response_dict}{Style.RESET_ALL}\n")
        tool_choice = agent_response_dict.get("tool_choice")
        tool_input = agent_response_dict.get("tool_input")

        for tool in self.tools:
            if tool.__name__ == tool_choice:
                response = tool(tool_input)
                print(f"{Fore.GREEN}Input argument: {tool_input}{Style.RESET_ALL}\n")
                print(f"{Fore.GREEN}Response: {response}{Style.RESET_ALL}\n")
                return response
        return None


if __name__ == "__main__":

    tools = [basic_calculator, parse_datetime]

    agent = Agent(tools=tools, model_name="gemini-2.0-flash-exp")
    prompt = "convert '2025-01-01' to a datetime object"

    agent.execute_planned_action(prompt)
