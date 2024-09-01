import random
from typing import Dict, Any
import asyncio
import math

from mctsmodels import MCTSNode, MCTSVisualizer, process_answer
from utils import *
from orchestrate import evaluate_simulation, expand_step, evaluate_answer, get_initial_steps, probe_step, ask_llama


async def mcts(question: str, max_iterations: int = 12, max_depth: int = 4) -> Dict[str, Any]:
    root = MCTSNode("root", node_id=0)
    initial_steps, estimated_total_steps = await get_initial_steps(question)
    next_node_id = 1  # Initialize the next node ID
    max_depth = estimated_total_steps
    for step in initial_steps:
        step.parent = root
        step.id = next_node_id  # Update the node id for each initial step
        root.children.append(step)
        next_node_id += 1
    visualizer = MCTSVisualizer()


    print(f"Starting MCTS with max_iterations={max_iterations} and max_depth={max_depth}")

    rejection_history = []  

    for iteration in range(max_iterations):
        log_metric("iterations", 1)
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Selection and Expansion
        node = root
        path = [root]
        

        if root.children:
            unexplored_children = [child for child in root.children if child.visits == 0]
            if unexplored_children:
                node = random.choice(unexplored_children)
            else:
                node = random.choice(root.children)
            path.append(node)
        while node.depth < max_depth:
            if not node.children:
                log_metric("expansion step, depth is:", node.depth)
                #sub_steps = await expand_step(node, question, max_depth)
                sub_steps = await probe_step(node, question, max_depth)
                for sub_step in sub_steps:
                    new_node = MCTSNode(sub_step, parent=node, node_id=next_node_id)
                    next_node_id += 1
                    node.children.append(new_node)
            
            # Select a child node
            if node.children:
                node = max(node.children, key=lambda n: ucb1_score(n, node.visits))
                path.append(node)
            else:
                break  # No valid sub-steps, stop expansion
        visualizer.update_graph(root, path)
        # Simulation
        print(f"Evaluating the following path: {path[-1].id}")
        answer, reason = await evaluate_simulation(node, question, path, rejection_history)
        is_valid, explanation = await evaluate_answer(answer, reason, question, path, node)
        if is_valid:
            return process_answer(answer, reason, path, visualizer)
        else:
            print(f"Answer rejected by Prometheus. Answer: {answer}, Reason: {explanation}")
            rejection_history.append({
                "Answer": answer,
                "Reason": reason,
                "Rejection": explanation
            }) 
            score = -1

        
        # Backpropagation
        for n in reversed(path):
            n.visits += 1
            n.value += score



    print("\nNo valid answer found within the maximum iterations")
    image = visualizer.create_image()
    return {"answer": "Unable to find a satisfactory answer within the given iterations.", 
            "reason": "The evaluation process did not pass for any of the generated answers.",
            "path": [{"id": node.id, "state": node.state if isinstance(node.state, str) else node.state.state, "parent_id": node.parent.id if node.parent else None} for node in path if node.state != "root"],
            "image": image}


def ucb1_score(node: MCTSNode, parent_visits: int) -> float:
    if node.visits == 0:
        return float('inf')  # Ensures unvisited nodes are explored first
    return (node.value / node.visits) + math.sqrt(2 * math.log(parent_visits) / node.visits)


async def main():
    log_metric("mcts_start", 1)
    print("MCTS process started")
    
    with open('questions.json', 'r') as f:
        questions_data = json.load(f)
    
    results = []

    for sample in questions_data['samples']:
        question_id = sample['id']
        context = sample['context']
        question = sample['question']
        choices = sample['choices']
        correct_answer = sample['answer']

        print(f"\nProcessing question {question_id}")

        # Ask llama3.1:8b
        llama_answer = await ask_llama(question, context, choices)
        print(f"llama3.1:8b answer: {llama_answer}")

        # Run MCTS
        combined_input = f"Context: {context}\n\nQuestion: {question}\n\nChoices:\n choose only one statement."
        for key, value in choices.items():
            combined_input += f"{key}: {value}\n"
        mcts_result = await mcts(combined_input)
        grapes_answer = mcts_result['answer']
        print(f"grapes_llama3.1:8b answer: {grapes_answer}")

        results.append({
            "id": question_id,
            "llama3.1:8b": llama_answer,
            "grapes_llama3.1:8b": grapes_answer,
            "correct_answer": correct_answer
        })

    # Save results to JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nAll questions processed. Results saved in 'results.json'")

    log_metric("mcts_end", 1)
    print("MCTS process ended")

if __name__ == "__main__":
    asyncio.run(main())
