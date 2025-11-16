# Batch LLM Processor

A Python tool for batch processing jobs using Anthropic's Batch API with tool use support, automatic state management, and multi-turn conversations.

## Features

- **Tool Use Support**: Define custom tools that Claude can call during processing
- **Multi-turn Conversations**: Automatically handles tool execution and continuation
- **Automatic Resumability**: If interrupted (CTRL+C), run the same command again to resume
- **Skip Completed Jobs**: Only processes jobs that don't have results yet
- **State Tracking**: Maintains `.batch_state.json` to track in-progress batches
- **Detailed Logging**: Stores full conversation history with tool executions in `results/logs/`
- **Shell Completion**: Tab completion for commands and options (bash, zsh, fish, PowerShell)


### Limitations

** If you run into an error because you hit your API limit, the job isn't easy to be restarted.

## Setup

1. Activate the virtual environment:
   ```bash
   # On Linux/macOS
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

4. (Optional) Enable shell completion:
   ```bash
   python process.py --install-completion
   # Then restart your shell
   ```

## Usage

### Folder Structure

Create a folder with the following structure:

```
my-job-folder/
├── SYSTEM_PROMPT.txt          # System prompt for all jobs
├── jobs/                      # Input jobs (*.txt files)
│   ├── 1.txt
│   ├── 2.txt
│   └── 3.txt
├── tools/                     # Optional: Tool definitions
│   └── my_tools.py            # Python file with @beta_tool decorated functions
└── results/                   # Output (created automatically)
    ├── 1.txt                  # Final response text
    ├── 2.txt
    └── logs/                  # Full conversation logs (array of turns)
        ├── 1.json
        └── 2.json
```

### Basic Usage

Process all jobs in a folder:
```bash
python process.py examples/tic-tac-toe
```

Specify a model (autocomplete shows short names):
```bash
# Use short names (with tab completion)
python process.py examples/calculator-tasks -m sonnet-4-5
python process.py examples/calculator-tasks -m opus-4
python process.py examples/calculator-tasks -m haiku-3-5  # Default

# Full model IDs also work
python process.py examples/calculator-tasks -m claude-sonnet-4-5-20250929
python process.py examples/calculator-tasks -m claude-3-5-haiku-20241022
```

Limit to first N jobs (useful for testing):
```bash
# Using --max-jobs
python process.py examples/calculator-tasks --max-jobs 3

# Or shorthand -n
python process.py examples/calculator-tasks -n 3
```

Compare models by storing results separately:
```bash
# Results go to results/haiku-3-5/
python process.py examples/calculator-tasks -m haiku-3-5 --model-result-dir

# Results go to results/sonnet-4-5/
python process.py examples/calculator-tasks -m sonnet-4-5 --model-result-dir

# Now you can compare results across models!
```

Get help:
```bash
python process.py --help
```

### Tool Use Example

See `examples/calculator-tasks/` for a complete example with tool use:

**tools/calculator.py:**
```python
from anthropic import beta_tool

@beta_tool
def add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: The first number to add
        b: The second number to add
        
    Returns:
        The sum of a and b
    """
    return str(a + b)

TOOLS = [add]  # Export tools
```

The processor will:
1. Load tools from `tools/*.py`
2. Include them in batch requests
3. Execute tool calls locally between batches
4. Continue conversations until completion

## Supported Models

All models supported by Anthropic's Batch API with autocomplete:

### Active Models
- **opus-4-1** → `claude-opus-4-1-20250805`
- **opus-4** → `claude-opus-4-20250514`
- **sonnet-4-5** → `claude-sonnet-4-5-20250929`
- **sonnet-4** → `claude-sonnet-4-20250514`
- **sonnet-3-7** → `claude-3-7-sonnet-20250219`
- **sonnet-3-5-v2** → `claude-3-5-sonnet-20241022`
- **haiku-3-5** → `claude-3-5-haiku-20241022` ← **Default**
- **haiku-3** → `claude-3-haiku-20240307`

### Deprecated (still supported)
- **sonnet-3-5-v1** → `claude-3-5-sonnet-20240620`
- **opus-3** → `claude-3-opus-20240229`

**Short names** (left side) are shown in autocomplete and recommended for CLI use. Full model IDs (right side) are also accepted.

## Examples

- **examples/tic-tac-toe/**: Simple text processing without tools
- **examples/calculator-tasks/**: Math problems with calculator tool use

## Dependencies

- anthropic v0.69.0
- typer v0.19.2
