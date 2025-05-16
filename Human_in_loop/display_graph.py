from typing import Optional
import os
from IPython.display import display
from graphviz import Digraph


def display_graph(graph, file_name: Optional[str] = None, format: str = "png", view: bool = False):
    """
    Display and save a visualization of a LangGraph StateGraph.

    Args:
        graph: The compiled LangGraph to visualize
        file_name: Optional name for the saved file (without extension)
        format: Output format for the graph image (default: "png")
        view: Whether to open the rendered graph image (default: False)
    """
    # Get the graph's underlying structure
    dot = graph.get_graph()

    # If we're in a Jupyter environment, display the graph
    try:
        display(dot)
    except:
        pass

    # Save the graph to file if file_name is provided
    if file_name:
        output_name = file_name.split('.')[0] if '.' in file_name else file_name
        dot.render(f"graph_{output_name}", format=format, cleanup=True, view=view)
        print(f"Graph saved as graph_{output_name}.{format}")

    return dot