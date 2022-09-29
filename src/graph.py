import networkx as nx
import metis
import matplotlib.pyplot as plt

def partition_graph(graph, n_parts):
    """
    Partition a graph into n_parts parts using METIS.
    """
    # Create a METIS graph from the networkx graph
    metis_graph = nx.to_metis(graph)
    # Partition the graph
    _, parts = metis.part_graph(metis_graph, n_parts)
    # Return the parts
    return parts

node = 1000

G = nx.complete_graph(node)
plt.cla()
nx.draw_networkx(G)
plt.savefig('graph.png')
plt.close()