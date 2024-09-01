
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
import asyncio
from ollama_instructor.ollama_instructor_client import OllamaInstructorAsyncClient

from mctsmodels import MCTSNode
from models import InitialSteps, SubSteps, PrometheusResponse, AnswerResponse, DepthEstimate, ProbingQuestion, LlamaResponse
from utils import *


async def ask_llama(question: str, context: str, choices: Dict[str, str]) -> str:
    client = OllamaInstructorAsyncClient()
    await client.async_init()

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nChoices:\n"
    for key, value in choices.items():
        prompt += f"{key}: {value}\n"
    prompt += "\nPlease select the most appropriate answer choice (e.g., choice_1, choice_2, etc.)."

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat_completion(
                model="llama3.1:8b",
                pydantic_model = LlamaResponse,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure")
            
            content = response['message']['content']
            if not content:
                raise ValueError("Empty response received")
            
            log_communication("llama", prompt, content, "ask_llama")
            return content['answer']
        
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Error in ask_llama (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default response.")
                return "Unable to generate a valid response"
            
            # Wait before retrying
            await asyncio.sleep(1)

async def ollama_generate(prompt: str, model: str = "llama3.1:8b", step_type: str = "generate", pydantic_model: BaseModel = None) -> Dict[str, Any]:
    client = OllamaInstructorAsyncClient()
    await client.async_init()

    with open("andromeda_prompt.txt", "r") as f:
        system_prompt = f.read().strip()
    

    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat_completion(
                model=model,
                pydantic_model=pydantic_model,
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{prompt}"},
                ],
            )
            
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure")
            
            content = response['message']['content']
            if not content:
                raise ValueError("Empty response received")
            
            log_communication(model, prompt, content, step_type)
            return content  # This should already be a dictionary
        
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Error in ollama_generate (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default response.")
                return {"error": "Failed to generate a valid response"}
            
            # Wait before retrying
            await asyncio.sleep(1)

async def prometheus_generate(prompt: str, pydantic_model: BaseModel = None) -> Dict[str, Any]:
    client = OllamaInstructorAsyncClient()
    await client.async_init()

    with open("prometheus_prompt.txt", "r") as f:
        system_prompt = f.read().strip()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat_completion(
                model="llama3.1:8b",
                pydantic_model=pydantic_model,
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{prompt}"},
                ],
            )
            
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure")
            
            content = response['message']['content']
            if not content:
                raise ValueError("Empty response received")
            
            log_communication("Prometheus", prompt, content, "generate")
            return content  # This should already be a dictionary
        
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Error in prometheus_generate (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default response.")
                return {"error": "Failed to generate a valid response"}
            
            # Wait before retrying
            await asyncio.sleep(1)


async def get_initial_steps(question: str) -> Tuple[List[MCTSNode], int]:
    prompt = f"Generate 3 possible first steps in the reasoning process to answer this question: {question}"
    response = await ollama_generate(prompt, pydantic_model=InitialSteps)
    nodes = []
    for step in response['steps']:
        nodes.append(MCTSNode(step['idea'], node_id=step['node_id']))

    prompt = f"Based on the question: {question}, estimate the depth of the reasoning process to answer the question. only output the depth estimate in the format of depth: integer"
    depth_estimate = await prometheus_generate(prompt, pydantic_model=DepthEstimate)
    return nodes, depth_estimate['depth']

async def expand_step(node: MCTSNode, question: str, max_depth: int) -> List[MCTSNode]:
    # Construct the full reasoning path
    path = []
    current = node
    while current:
        if current.state != 'root':
            if isinstance(current.state, str):
                path.append(current.state)
            else:
                path.append(current.state.state)
        current = current.parent
    path.reverse()  # Reverse to get the path from root to current node
    
    # Add step counter to each step
    numbered_path = [f"Step {i+1}: {step}" for i, step in enumerate(path)]
    path = numbered_path

    prompt = f"Provide the next 2 possible sub-steps in the reasoning process to answer the question.\nThe question is: '{question}'\nThe current path of logical reasoning is:\n{' -> '.join(path)}\n Expand this path with 2 possible next steps. \
    \n Remember your logical reasoning steps should be at most 5 steps long, you are at step {node.depth+1} of {max_depth}."
    response = await ollama_generate(prompt, pydantic_model=SubSteps)
    sub_steps = []
    for step in response['sub_steps']:
        new_node = MCTSNode(step['idea'], parent=node)
        sub_steps.append(new_node)
    return sub_steps

async def probe_step(node: MCTSNode, question: str, max_depth: int) -> List[MCTSNode]:
    # Construct the full reasoning path
    path = []
    current = node
    while current:
        if current.state != 'root':
            if isinstance(current.state, str):
                path.append(current.state)
            else:
                path.append(current.state.state)
        current = current.parent
    path.reverse()  # Reverse to get the path from root to current node
    
    # Add step counter to each step
    numbered_path = [f"Step {i+1}: {step}" for i, step in enumerate(path)]
    path = numbered_path

    #Prometheus generates probing questions
    probe_prompt = f"Based on the question '{question}' and the current reasoning path:\n{' -> '.join(path)}\nGenerate a probing question that could lead to insightful next steps in the reasoning process. output a single JSON object with the probing_question key."
    probe_response = await prometheus_generate(probe_prompt, pydantic_model=ProbingQuestion)
    if 'error' in probe_response:
        probing_question = "What would you do next to get to the answer?"
    else:
        probing_question = probe_response['probing_question'][0]

    # Andromeda generates sub-steps based on Prometheus's probes
    prompt = f"Consider the following question: '{question}'\nCurrent reasoning path:\n{' -> '.join(path)}\n\nPrometheus asks:\n1. {probing_question}\n\n \
    Provide two possible next steps in the reasoning process, each addressing one of Prometheus's questions. \
    Remember, your logical reasoning steps should be at most {max_depth} steps long, and you are at step {node.depth+1} of {max_depth}."
    
    response = await ollama_generate(prompt, pydantic_model=SubSteps)
    sub_steps = []
    for step in response['sub_steps']:
        new_node = MCTSNode(step['idea'], parent=node)
        sub_steps.append(new_node)
    return sub_steps

async def evaluate_answer(answer: str, reason: str, question: str, path: List[MCTSNode], step: MCTSNode) -> Tuple[bool, str]:
    context = [node.state if isinstance(node.state, str) else node.state.state for node in path if node != step and node.state != 'root']
    context = "\n".join(context)
    prompt = f"Evaluate this answer and reason, focusing on inconsistencies or logical errors:\nAnswer: {answer}\nReason: {reason}\n Original question:{question}\n Is this answer logically consistent and correct? give a consise feedback, 3 short sentences maximum."
    response = await prometheus_generate(prompt, PrometheusResponse)
    is_valid = response['feedback'] == "PASS"
    explanation = response['explanation']
    log_metric("prometheus", is_valid)

    return is_valid, explanation


async def evaluate_simulation(step: MCTSNode, question: str, path: List[MCTSNode], rejection_history: List[str]) -> Tuple[str, str]:
    # Create context only from the current chain of nodes
    rejection_context = "\n".join(
        f"Previous attempt {i+1}:\nAnswer: {item['Answer']}\nReason: {item['Reason']}\nRejection: {item['Rejection']}"
        for i, item in enumerate(rejection_history)
    )
    context = [node.state if isinstance(node.state, str) else node.state.state for node in path if node != step and node.state != 'root']
    context_str = "\n".join(context)

    prompt = f"""Given the question: '{question}'
AND the following context of  explored steps in the current chain:
{context_str}

Previous rejections, take this into account very seriously. do not make the same mistake or give the same wrong answer: \n \
{rejection_context}

give me an answer in the specific format of:
'''Answer: this is the final answer based on the choices, Reason: section after the answer, explaining the logic behind the answer based on the explored steps. 3-4 short sentences maximum. '''
"""
    response = await ollama_generate(prompt, model="llama3.1:8b", step_type="evaluate", pydantic_model=AnswerResponse)
    answer = response['Answer']
    reason = response['Reason']

    return answer, reason