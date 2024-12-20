import sys

sys.path.insert(0, "..")
from colorama import init, Fore, Style
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
    def __init__(self, tools, model_service, model_name):
        """
        Initializes the agent with a list of tools and a model.

        Parameters:
        tools (list): List of custom functions as tools.
        model_service (class): The model service class with a generate_text method.
        model_name (str): The name of the model to use.
        """
        self.tools = tools
        self.model_service = GeminiModel
        self.model_name = model_name

    def prepare_tools(self):
        """
        Stores the tools in the toolbox and returns their descriptions.

        Returns:
        str: Descriptions of the tools stored in the toolbox.
        """
        toolbox = ToolBox()
        toolbox.register_functions(self.tools)
        tool_descriptions = toolbox.get_registered_functions_as_string()
        return tool_descriptions

    def think(self, prompt):
        """
        Runs the generate_text method on the model using the system prompt template and tool descriptions.

        Parameters:
        prompt (str): The user query to generate a response for.

        Returns:
        dict: The response from the model as a dictionary.
        """
        tool_descriptions = self.prepare_tools()
        agent_system_prompt = agent_system_prompt_template.format(
            tool_descriptions=tool_descriptions
        )

        logger.info(
            f"{Fore.CYAN}System prompt: {agent_system_prompt} {Style.RESET_ALL}"
        )

        model_instance = GeminiModel(
            model_name=self.model_name, system_prompt=agent_system_prompt, temperature=0
        )

        # Generate and return the response dictionary
        agent_response_dict = model_instance.generate_text(prompt)
        return agent_response_dict

    def execute(self, prompt):
        """
        Parses the dictionary returned from think and executes the appropriate tool.

        Parameters:
        prompt (str): The user query to generate a response for.

        Returns:
        The response from executing the appropriate tool or the tool_input if no matching tool is found.
        """
        agent_response_dict = self.think(prompt)
        tool_choice = agent_response_dict.get("tool_choice")
        tool_input = agent_response_dict.get("tool_input")

        for tool in self.tools:
            if tool.__name__ == tool_choice:
                response = tool(tool_input)
                print(f"{Fore.GREEN}Input argument: {tool_input}{Style.RESET_ALL}\n")
                print(f"{Fore.GREEN}Response: {response}{Style.RESET_ALL}\n")
                return
        return


if __name__ == "__main__":

    tools = [basic_calculator, parse_datetime]

    agent = Agent(
        tools=tools, model_service=GeminiModel, model_name="gemini-2.0-flash-exp"
    )
    prompt = "convert '2025-01-01' to a datetime object"

    agent.execute(prompt)
