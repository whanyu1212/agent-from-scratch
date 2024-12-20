# Agent From Scratch

A lightweight LLM agent implementation using Google's Gemini model, built without intermediate libraries. This project demonstrates how to create an AI agent that can use tools and respond to queries directly.

### How It Works

1. **User input analysis**
   - Agent receives user input
   - Analyzes query using Gemini model
   - Determines if tools are needed

<br>

2. **Decision making and response output**
   - If query matches tool capability:
     - Selects appropriate tool
     - Formats tool inputs
     - Executes tool function
   - If no tool needed:
     - Generates direct response using LLM


### Configuring environment

```bash
# Clone repository
git clone https://github.com/whanyu1212/agent-from-scratch.git
cd agent-from-scratch

# Syncing the dependencies listed
uv sync

# Activate the environment
source venv/bin/activate  # Linux/Mac

# or
.\venv\Scripts\activate  # Windows

# Set up environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```


### Usage
Please refer to [example notebook](notebook/example.ipynb) for a simple demonstration.