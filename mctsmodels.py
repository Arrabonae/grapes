from typing import List, Dict, Any, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class MCTSNode:
    def __init__(self, state: str, parent=None, node_id: int = None):
        self.id = node_id
        self.state = state
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.value = 0


class MCTSVisualizer:
    def __init__(self):
        self.G = nx.DiGraph()  # Use DiGraph for directed edges
        self.pos = None
        self.node_colors = []
        self.edge_colors = []
        self.node_labels = {}

    def update_graph(self, root: MCTSNode, highlighted_path: List[MCTSNode]):
        self.G.clear()  # Clear the graph before rebuilding
        self._add_nodes_recursive(root)

        self.pos = graphviz_layout(self.G, prog="dot", root=root.id)  # Use tree layout with root at the top
        
        highlighted_ids = [node.id for node in highlighted_path]
        self.node_colors = ['red' if node in highlighted_ids else 'lightblue' for node in self.G.nodes()]
        
        highlighted_edges = list(zip(highlighted_ids[:-1], highlighted_ids[1:]))
        self.edge_colors = ['red' if edge in highlighted_edges else 'gray' for edge in self.G.edges()]

    def _add_nodes_recursive(self, node: MCTSNode):
        self.G.add_node(node.id)
        self.node_labels[node.id] = str(node.id)
        
        for child in node.children:
            self.G.add_edge(node.id, child.id)
            self._add_nodes_recursive(child)

    def create_image(self):
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, self.pos, node_color=self.node_colors, edge_color=self.edge_colors, 
                with_labels=True, labels=self.node_labels,
                node_size=3000, font_size=8, font_weight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_str
    

def process_answer(answer: str, reason: str, path: List[MCTSNode], visualizer: MCTSVisualizer) -> Dict[str, Any]:

    image = visualizer.create_image()
    # Convert the path to a list of dictionaries containing node information
    path_info = [{"id": node.id, "state": node.state if isinstance(node.state, str) else node.state.state, "parent_id": node.parent.id if node.parent else None} for node in path if node.state != "root"]
    return {"answer": answer, "reason": reason, "path": path_info, "image": image}


def prune_path(root: MCTSNode, path: List[MCTSNode]):
    current = root
    for node in path[1:]:  # Skip the root node
        if node in current.children:
            current.children.remove(node)
        if not current.children:
            break
        current = current.children[0]