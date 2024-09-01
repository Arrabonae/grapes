# GRAPES: Graph-based Reasoning and Planning with Ensemble Systems

## Overview
GRAPES (Graph-based Reasoning and Planning with Ensemble Systems) is an innovative approach to enhancing language model reasoning capabilities through a Monte Carlo Tree Search (MCTS) based prompting mechanism. This system leverages a dual-model setup, utilizing Andromeda and Prometheus models, to improve reasoning outcomes significantly. The below system uses the open source llama3.1:8b model.

## Key Components

### 1. Monte Carlo Tree Search (MCTS)
At the core of GRAPES is the MCTS algorithm, adapted for language model reasoning:
- **Selection**: The algorithm starts from the root node and selects child nodes based on the UCB1 (Upper Confidence Bound 1) score, balancing exploration and exploitation.
- **Expansion**: When a leaf node is reached, it's expanded by generating potential next steps in the reasoning process. Generation done by probing questions from Prometheus model. 
- **Simulation**: The current path is evaluated to produce an answer and reason.
- **Backpropagation**: The evaluation results are propagated back up the tree, updating node statistics. Evaluation is done by the Prometheus model.

### 2. Dual Model Architecture
GRAPES employs two specialized models:
- **Andromeda**: Responsible for generating reasoning steps and potential answers.
- **Prometheus**: Evaluates the quality and logical consistency of the generated answers.

### 3. Graph Visualization
The system includes a `MCTSVisualizer` class that creates a visual representation of the search tree, enhancing interpretability.

## Key Algorithms and Techniques

### 1. UCB1 Score Calculation
The UCB1 score is used for node selection:
```python
def ucb1_score(node: MCTSNode, parent_visits: int) -> float:
    if node.visits == 0:
        return float('inf')
    return (node.value / node.visits) + math.sqrt(2 * math.log(parent_visits) / node.visits)
```

### 2. Probing Questions
Prometheus generates probing questions to guide the exploration:
```python
probe_prompt = f"Based on the question '{question}' and the current reasoning path:\n{' -> '.join(path)}\nGenerate a probing question that could lead to insightful next steps in the reasoning process."
```

### 3. Answer Evaluation
Prometheus evaluates the generated answers:
```python
prompt = f"Evaluate this answer and reason, focusing on inconsistencies or logical errors:\nAnswer: {answer}\nReason: {reason}\n Original question:{question}\n Is this answer logically consistent and correct?"
```

### 4. Rejection History
The system maintains a rejection history to avoid repeating mistakes:
```python
rejection_context = "\n".join(
    f"Previous attempt {i+1}:\nAnswer: {item['Answer']}\nReason: {item['Reason']}\nRejection: {item['Rejection']}"
    for i, item in enumerate(rejection_history)
)
```

## Implementation Details

1. **Asynchronous Processing**: The system uses `asyncio` for concurrent operations, enhancing efficiency.
2. **Pydantic Models**: Structured data models ensure type safety and validation.
3. **Configurable Parameters**: The MCTS process can be fine-tuned with parameters like `max_iterations` and `max_depth`.
4. **Error Handling**: Robust error handling with retries for model queries.

## Results

The results of the GRAPES model show significant improvement in reasoning capabilities:
0. **Dataset**:
    - models are tested on the following dataset: https://github.com/Mihir3009/LogicBench/tree/main/data/LogicBench(Eval)
    - questions and asnwers are a collection of 64 reasoning questions from the repository above. 

1. **Accuracy Improvement**: 
   - llama3.1:8b model: 62.50% accuracy
   - GRAPES (grapes_llama3.1:8b): 78.12% accuracy

2. **Reasoning Quality**: The GRAPES model demonstrated more coherent and logically consistent reasoning paths.

3. **Explainability**: The graph visualization provides insights into the reasoning process, enhancing the interpretability of the model's decisions. Also, the reasoning path is easily audited.

## Conclusion

GRAPES shows promising improvement in llama3.1:8b model reasoning capaibilities. By combining MCTS with a dual-model architecture, it achieves more accurate and logically consistent responses. The system's ability to learn from rejections and adapt its reasoning path showcases its potential for continual improvement in complex reasoning tasks.