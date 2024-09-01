import random
from typing import List, Dict, Any, Tuple
import ollama
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from enum import Enum
import rich
import asyncio
from ollama_instructor.ollama_instructor_client import OllamaInstructorAsyncClient
import math


def log_metric(metric_name: str, value: Any):
    print(f"Metric: {metric_name}, Value: {value}")

def log_communication(model: str, prompt: str, response: str, step_type: str):
    timestamp = datetime.now().isoformat()
    
    # Remove system prompt from the logged prompt
    user_prompt = prompt.split("User:", 1)[-1].strip()
    
    log_entry = {
        "timestamp": timestamp,
        "model": model,
        "prompt": user_prompt,
        "response": response,
        "step_type": step_type
    }
    filename = f"{model}_communication_log.txt"
    with open(filename, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")