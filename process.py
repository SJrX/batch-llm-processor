#!/usr/bin/env python3
"""
Batch LLM Processor - Phase 2
Submits jobs to Anthropic's Batch API with tool use support.
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import typer
from anthropic import Anthropic


class ClaudeModel(str, Enum):
    """Supported Claude models for Batch API (short names)"""
    opus_4_1 = "opus-4-1"
    opus_4 = "opus-4"
    sonnet_4_5 = "sonnet-4-5"
    sonnet_4 = "sonnet-4"
    sonnet_3_7 = "sonnet-3-7"
    sonnet_3_5_v2 = "sonnet-3-5-v2"
    sonnet_3_5_v1 = "sonnet-3-5-v1"
    haiku_3_5 = "haiku-3-5"
    haiku_3 = "haiku-3"
    opus_3 = "opus-3"


# Mapping from short names to full model IDs
MODEL_FULL_IDS = {
    "opus-4-1": "claude-opus-4-1-20250805",
    "opus-4": "claude-opus-4-20250514",
    "sonnet-4-5": "claude-sonnet-4-5-20250929",
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-3-7": "claude-3-7-sonnet-20250219",
    "sonnet-3-5-v2": "claude-3-5-sonnet-20241022",
    "sonnet-3-5-v1": "claude-3-5-sonnet-20240620",
    "haiku-3-5": "claude-3-5-haiku-20241022",
    "haiku-3": "claude-3-haiku-20240307",
    "opus-3": "claude-3-opus-20240229",
}


def resolve_model_id(model_name: str) -> str:
    """Convert short model name to full ID, or pass through if already full ID"""
    # If it's already a full ID (starts with "claude-"), use it directly
    if model_name.startswith("claude-"):
        return model_name
    # Otherwise map the short name to full ID
    return MODEL_FULL_IDS.get(model_name, model_name)


# Configuration
MAX_TOKENS = 4096
POLL_INTERVAL_SECONDS = 10
STATE_FILE = ".batch_state.json"


class BatchProcessor:
    def __init__(self, folder: Path, model: str = "claude-3-5-haiku-20241022", max_jobs: Optional[int] = None, model_result_dir: bool = False):
        self.folder = folder
        self.model = model  # Should be full model ID
        self.max_jobs = max_jobs
        self.jobs_dir = folder / "jobs"

        # Use model-specific results dir if requested
        if model_result_dir:
            # Extract short model name (e.g., "haiku-3-5" from "claude-3-5-haiku-20241022")
            model_name = model.replace("claude-", "").rsplit("-", 1)[0]
            self.results_dir = folder / "results" / model_name
        else:
            self.results_dir = folder / "results"

        self.logs_dir = self.results_dir / "logs"
        self.tools_dir = folder / "tools"
        self.state_file = folder / STATE_FILE
        self.system_prompt_file = folder / "SYSTEM_PROMPT.txt"

        # Initialize Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)

        self.client = Anthropic(api_key=api_key)

        # Initialize tool storage before loading
        self.tool_functions = {}
        self.tools = self.load_tools()

        # Initialize job ID mapping for long filenames
        self.job_id_map = {}  # Maps short_id -> original filename
        self.reverse_job_id_map = {}  # Maps original filename -> short_id

        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)

    def load_tools(self) -> List[Any]:
        """Load tools from the tools/ directory"""
        if not self.tools_dir.exists():
            return []

        tools = []
        tool_files = list(self.tools_dir.glob("*.py"))

        if not tool_files:
            return []

        print(f"Loading tools from {self.tools_dir}...")

        for tool_file in tool_files:
            if tool_file.name.startswith("__"):
                continue

            # Load the module
            spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Get TOOLS list from module
                if hasattr(module, "TOOLS"):
                    module_tools = module.TOOLS
                    tools.extend(module_tools)

                    # Store tool functions for execution
                    for tool in module_tools:
                        # @beta_tool creates a wrapper, get the name from the tool definition
                        tool_name = None
                        if hasattr(tool, '__name__'):
                            tool_name = tool.__name__
                        elif hasattr(tool, 'name'):
                            tool_name = tool.name
                        elif hasattr(tool, 'to_dict') and callable(tool.to_dict):
                            # Get name from the tool schema
                            tool_dict = tool.to_dict()
                            tool_name = tool_dict.get('name')

                        if tool_name:
                            self.tool_functions[tool_name] = tool
                            print(f"    Registered tool: {tool_name}")
                        else:
                            print(f"    Warning: Could not determine name for tool: {tool}")

                    print(f"  Loaded {len(module_tools)} tools from {tool_file.name}")

        print(f"Total tool functions registered: {list(self.tool_functions.keys())}")
        return tools

    def load_system_prompt(self) -> str:
        """Load the system prompt from SYSTEM_PROMPT.txt"""
        if not self.system_prompt_file.exists():
            print(f"Error: {self.system_prompt_file} not found")
            sys.exit(1)

        return self.system_prompt_file.read_text().strip()

    def get_pending_jobs(self) -> List[Path]:
        """Get list of jobs that don't have results yet"""
        if not self.jobs_dir.exists():
            print(f"Error: {self.jobs_dir} not found")
            sys.exit(1)

        job_files = sorted(self.jobs_dir.glob("*"))
        pending = []

        for job_file in job_files:
            result_file = self.results_dir / job_file.name
            if not result_file.exists():
                pending.append(job_file)

        return pending

    def load_state(self) -> Optional[Dict]:
        """Load the current batch state"""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return None

    def save_state(self, batch_id: str, job_files: List[str], conversations: Optional[Dict] = None):
        """Save the current batch state"""
        state = {
            "batch_id": batch_id,
            "job_files": job_files,
            "conversations": conversations or {},
            "timestamp": time.time()
        }
        self.state_file.write_text(json.dumps(state, indent=2))
        print(f"State saved to: {self.state_file}")

    def clear_state(self):
        """Clear the state file"""
        if self.state_file.exists():
            self.state_file.unlink()

    def get_short_id(self, filename: str) -> str:
        """Generate a short ID for a filename (max 64 chars for API)"""
        # If already mapped, return existing ID
        if filename in self.reverse_job_id_map:
            return self.reverse_job_id_map[filename]

        # Sanitize filename: replace invalid chars with underscore
        # API allows only: [a-zA-Z0-9_-]
        sanitized = filename.replace('.', '_').replace(' ', '_')
        # Remove any other invalid characters
        sanitized = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in sanitized)

        # If sanitized name is short enough, use it directly
        if len(sanitized) <= 60:  # Leave room for safety
            short_id = sanitized
        else:
            # For long filenames, use first 50 chars + hash of full name
            file_hash = hashlib.sha256(filename.encode()).hexdigest()[:10]
            short_id = f"{sanitized[:50]}_{file_hash}"

        # Store mapping
        self.job_id_map[short_id] = filename
        self.reverse_job_id_map[filename] = short_id
        return short_id

    def get_original_filename(self, short_id: str) -> str:
        """Get original filename from short ID"""
        return self.job_id_map.get(short_id, short_id)

    def create_batch_requests(self, job_files: List[Path], system_prompt: str, conversations: Optional[Dict] = None) -> List[Dict]:
        """Create batch request objects for the API"""
        requests = []
        conversations = conversations or {}

        for job_file in job_files:
            # Use name (with extension) as base, then shorten if needed
            filename = job_file.name
            job_id = self.get_short_id(filename)

            # Use existing conversation or start new one
            if job_id in conversations:
                messages = conversations[job_id]
            else:
                user_message = job_file.read_text().strip()
                messages = [{"role": "user", "content": user_message}]

            request = {
                "custom_id": job_id,
                "params": {
                    "model": self.model,
                    "max_tokens": MAX_TOKENS,
                    "messages": messages
                }
            }

            # Add system prompt if provided
            if system_prompt:
                request["params"]["system"] = system_prompt

            # Add tools if available
            if self.tools:
                # Convert tool objects to dicts
                request["params"]["tools"] = [tool.to_dict() if hasattr(tool, 'to_dict') else tool for tool in self.tools]

            requests.append(request)

        return requests

    def submit_batch(self, requests: List[Dict]) -> str:
        """Submit a batch to the API and return the batch ID"""
        print(f"Submitting batch with {len(requests)} requests...")

        # Create the batch
        batch = self.client.messages.batches.create(requests=requests)

        print(f"Batch created: {batch.id}")
        print(f"Status: {batch.processing_status}")

        return batch.id

    def poll_batch(self, batch_id: str) -> Dict:
        """Poll the batch until completion or error"""
        print(f"Polling batch {batch_id}...")

        first_poll = True
        while True:
            try:
                batch = self.client.messages.batches.retrieve(batch_id)
                status = batch.processing_status
                counts = batch.request_counts

                # Show batch timing info on first poll
                if first_poll:
                    print(f"Created: {batch.created_at}")
                    print(f"Expires: {batch.expires_at}")
                    if batch.ended_at:
                        print(f"Ended: {batch.ended_at}")
                    first_poll = False

                # Show processing progress
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                completed = counts.succeeded + counts.errored + counts.canceled + counts.expired
                print(f"Status: {status} | Completed: {completed}/{total} | Succeeded: {counts.succeeded} | Errored: {counts.errored}")

                if status == "ended":
                    print("Batch completed!")
                    return batch
                elif status in ["canceled", "expired"]:
                    print(f"Batch {status}")
                    return batch

                time.sleep(POLL_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                print("\nInterrupted. State saved. Run again to resume.")
                sys.exit(0)

    def execute_tool(self, tool_name: str, tool_input: dict, job_id: str = None) -> str:
        """Execute a tool by name with given input

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            job_id: Optional job ID to expose to the tool via environment variable
        """
        if tool_name not in self.tool_functions:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            # Change to job folder so tools can use relative paths
            original_cwd = os.getcwd()
            original_job_id = os.environ.get('BATCH_JOB_ID')
            original_results_dir = os.environ.get('BATCH_RESULTS_DIR')

            os.chdir(self.folder)

            # Set job ID in environment for tools to optionally use
            if job_id:
                os.environ['BATCH_JOB_ID'] = job_id

            # Set results directory path (relative to job folder)
            results_dir_relative = self.results_dir.relative_to(self.folder)
            os.environ['BATCH_RESULTS_DIR'] = str(results_dir_relative)

            # Set job file path (absolute) - for tools that need to read the original prompt
            if job_id:
                original_filename = self.get_original_filename(job_id)
                job_file_path = self.jobs_dir / original_filename
                os.environ['BATCH_JOB_FILE'] = str(job_file_path.absolute())

            try:
                tool_func = self.tool_functions[tool_name]
                result = tool_func(**tool_input)
                return result
            finally:
                # Always restore original directory and environment
                os.chdir(original_cwd)
                if original_job_id is not None:
                    os.environ['BATCH_JOB_ID'] = original_job_id
                elif 'BATCH_JOB_ID' in os.environ:
                    del os.environ['BATCH_JOB_ID']

                if original_results_dir is not None:
                    os.environ['BATCH_RESULTS_DIR'] = original_results_dir
                elif 'BATCH_RESULTS_DIR' in os.environ:
                    del os.environ['BATCH_RESULTS_DIR']

                if 'BATCH_JOB_FILE' in os.environ:
                    del os.environ['BATCH_JOB_FILE']
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def process_batch_results(self, batch_id: str, turn: int) -> Dict[str, Any]:
        """Process batch results and categorize by completion status"""
        print("Retrieving results...")

        results = self.client.messages.batches.results(batch_id)

        completed = {}  # Jobs that reached end_turn
        needs_tool_execution = {}  # Jobs that need tool execution
        errored = {}  # Jobs that errored

        for result in results:
            custom_id = result.custom_id
            # Map back to original filename for logging
            original_filename = self.get_original_filename(custom_id)
            log_file = self.logs_dir / f"{original_filename}.json"

            # Append to log array (track all turns)
            log_entry = {
                "turn": turn,
                "batch_id": batch_id,
                "result": result.model_dump()
            }

            # Read existing log or create new array
            if log_file.exists():
                existing_logs = json.loads(log_file.read_text())
            else:
                existing_logs = []

            existing_logs.append(log_entry)
            log_file.write_text(json.dumps(existing_logs, indent=2))

            if result.result.type == "succeeded":
                message = result.result.message
                stop_reason = message.stop_reason

                if stop_reason == "end_turn":
                    # Final answer - extract text
                    text_content = []
                    for block in message.content:
                        if block.type == "text":
                            text_content.append(block.text)
                    completed[custom_id] = "\n".join(text_content)

                elif stop_reason == "tool_use":
                    # Needs tool execution
                    needs_tool_execution[custom_id] = message

            else:
                # Error
                error_info = result.result.error
                error_text = f"Error Type: {result.result.type}\n"
                error_text += f"Error: {json.dumps(error_info.model_dump() if hasattr(error_info, 'model_dump') else str(error_info), indent=2)}"
                errored[custom_id] = error_text

        return {
            "completed": completed,
            "needs_tool_execution": needs_tool_execution,
            "errored": errored
        }

    def write_final_results(self, completed: Dict[str, str], errored: Dict[str, str]):
        """Write final results to disk"""
        for custom_id, text in completed.items():
            original_filename = self.get_original_filename(custom_id)
            result_file = self.results_dir / original_filename
            result_file.write_text(text)
            print(f"✓ {original_filename}")

        for custom_id, error_text in errored.items():
            original_filename = self.get_original_filename(custom_id)
            result_file = self.results_dir / original_filename
            result_file.write_text(error_text)
            print(f"✗ {original_filename} (error)")

        if completed or errored:
            print(f"\nResults written to {self.results_dir}")

    def create_tool_result_conversations(self, needs_tool_execution: Dict, conversations: Dict, turn: int) -> Dict:
        """Execute tools and create continuation conversations"""
        print(f"\nExecuting tools for {len(needs_tool_execution)} jobs...")

        new_conversations = {}

        for custom_id, message in needs_tool_execution.items():
            # Get existing conversation or start new
            if custom_id in conversations:
                conv = conversations[custom_id].copy()
            else:
                conv = []

            # Add assistant message with tool calls
            conv.append({
                "role": "assistant",
                "content": [block.model_dump() for block in message.content]
            })

            # Execute tools and collect results
            tool_results = []
            tool_executions = []  # For logging

            for block in message.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"  {custom_id}: Executing {tool_name}({tool_input})")

                    try:
                        result = self.execute_tool(tool_name, tool_input, job_id=custom_id)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result
                        })
                        tool_executions.append({
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "result": result,
                            "error": False
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(e),
                            "is_error": True
                        })
                        tool_executions.append({
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "error": str(e)
                        })

            # Log tool executions
            original_filename = self.get_original_filename(custom_id)
            log_file = self.logs_dir / f"{original_filename}.json"
            if log_file.exists():
                existing_logs = json.loads(log_file.read_text())
                existing_logs.append({
                    "turn": turn,
                    "type": "tool_execution",
                    "executions": tool_executions
                })
                log_file.write_text(json.dumps(existing_logs, indent=2))

            # Add user message with tool results
            conv.append({
                "role": "user",
                "content": tool_results
            })

            new_conversations[custom_id] = conv

        return new_conversations

    def run(self):
        """Main execution flow with multi-turn tool support"""
        start_time = time.time()
        
        print(f"Batch LLM Processor")
        print(f"Folder: {self.folder}")
        print(f"Model: {self.model}")
        print("-" * 50)

        if self.tools:
            print(f"Tools enabled: {len(self.tools)} tools loaded")

        # Load system prompt
        system_prompt = self.load_system_prompt()
        print(f"System prompt loaded: {len(system_prompt)} chars")

        # Get pending jobs and build ID mappings
        pending_jobs = self.get_pending_jobs()

        if not pending_jobs:
            print("No pending jobs found. All jobs have results.")
            return

        # Pre-build job ID mappings for all pending jobs
        for job_file in pending_jobs:
            self.get_short_id(job_file.name)

        # Limit jobs if max_jobs is set
        if self.max_jobs and len(pending_jobs) > self.max_jobs:
            print(f"Found {len(pending_jobs)} pending jobs, limiting to {self.max_jobs}")
            pending_jobs = pending_jobs[:self.max_jobs]
        else:
            print(f"Found {len(pending_jobs)} pending jobs")

        # Track conversations for multi-turn
        conversations = {}
        all_completed = {}
        all_errored = {}

        # Multi-turn loop
        max_turns = 10  # Safety limit
        turn = 0

        while pending_jobs and turn < max_turns:
            turn += 1
            print(f"\n=== Turn {turn} ===")

            # Check for existing batch in progress
            state = self.load_state()

            if state and state["batch_id"] != "pending":
                batch_id = state["batch_id"]
                job_files = state["job_files"]
                conversations = state.get("conversations", {})
                print(f"Found existing batch: {batch_id}")

                # Poll the batch
                batch = self.poll_batch(batch_id)

                if batch.processing_status != "ended":
                    print("Batch not completed yet")
                    return
            else:
                # If state exists with "pending" batch_id, restore conversations
                if state and state["batch_id"] == "pending":
                    conversations = state.get("conversations", {})
                    print(f"Resuming from saved state (turn {turn})")


                # Create batch requests
                requests = self.create_batch_requests(pending_jobs, system_prompt, conversations)

                # Submit batch
                batch_id = self.submit_batch(requests)

                # Save state
                job_filenames = [job.name for job in pending_jobs]
                self.save_state(batch_id, job_filenames, conversations)

                # Poll for completion
                batch = self.poll_batch(batch_id)

                if batch.processing_status != "ended":
                    print("Batch not completed yet")
                    return

            # Process results
            results = self.process_batch_results(batch_id, turn)

            # Add completed and errored jobs to final results
            all_completed.update(results["completed"])
            all_errored.update(results["errored"])

            print(f"\nCompleted: {len(results['completed'])} | Needs tools: {len(results['needs_tool_execution'])} | Errors: {len(results['errored'])}")

            # If no more tool executions needed, we're done
            if not results["needs_tool_execution"]:
                elapsed_time = time.time() - start_time
                print("\nAll jobs completed!")
                self.write_final_results(all_completed, all_errored)
                self.clear_state()
                self._print_elapsed_time(elapsed_time)
                break

            # Execute tools and create continuation conversations
            conversations = self.create_tool_result_conversations(
                results["needs_tool_execution"],
                conversations,
                turn
            )

            # Update pending jobs to only those needing continuation
            pending_job_ids = set(results["needs_tool_execution"].keys())
            pending_jobs = [j for j in pending_jobs if self.get_short_id(j.name) in pending_job_ids]

            # Save state with updated conversations before clearing
            # This ensures we can resume if interrupted between turns
            if pending_jobs:
                job_filenames = [job.name for job in pending_jobs]
                # Use a placeholder batch_id - will be updated when next batch is created
                self.save_state("pending", job_filenames, conversations)

        if turn >= max_turns:
            elapsed_time = time.time() - start_time
            print(f"\n⚠ Reached maximum turns ({max_turns}). Some jobs may not be complete.")
            self.write_final_results(all_completed, all_errored)
            self._print_elapsed_time(elapsed_time)

    def _print_elapsed_time(self, elapsed_seconds: float):
        """Print formatted elapsed time"""
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        time_str = " ".join(parts)
        print(f"\n⏱  Total time: {time_str} ({elapsed_seconds:.1f}s)")


app = typer.Typer(help="Batch LLM Processor - Process jobs using Anthropic's Batch API with tool support")


@app.command()
def process(
    folder: str = typer.Argument(
        help="Path to the job folder containing jobs/, SYSTEM_PROMPT.txt, and optionally tools/"
    ),
    model: ClaudeModel = typer.Option(
        ClaudeModel.haiku_3_5,
        "--model",
        "-m",
        help="Claude model to use for processing",
        case_sensitive=False
    ),
    max_jobs: Optional[int] = typer.Option(
        None,
        "--max-jobs",
        "-n",
        help="Maximum number of jobs to process (useful for testing)"
    ),
    model_result_dir: bool = typer.Option(
        False,
        "--model-result-dir",
        help="Store results in results/<model>/ instead of results/"
    ),
):
    """
    Process batch jobs with the Anthropic API.

    This will:
    1. Load jobs from the jobs/ directory
    2. Load tools from tools/ directory (if present)
    3. Submit jobs as a batch to Anthropic
    4. Handle multi-turn conversations with tool execution
    5. Write final results to results/ or results/<model>/ directory
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        typer.echo(f"Error: Folder {folder} does not exist", err=True)
        raise typer.Exit(1)

    if max_jobs is not None and max_jobs < 1:
        typer.echo(f"Error: --max-jobs must be at least 1", err=True)
        raise typer.Exit(1)

    # Resolve short model name to full ID
    model_id = resolve_model_id(model.value)

    processor = BatchProcessor(
        folder_path,
        model=model_id,
        max_jobs=max_jobs,
        model_result_dir=model_result_dir
    )
    processor.run()


if __name__ == "__main__":
    app()
