import networkx as nx
import metis
import matplotlib.pyplot as plt
from ortoolpy import networkx_draw

node = 12
G = nx.complete_graph(node)
Ga = nx.to_directed(G)
nx.draw(Ga, with_labels=True)
plt.savefig('graph.png', dpi=300)

npart = 3
(edgecut, parts) = metis.part_graph(G, npart)
print("edgecut: {}".format(edgecut))
print("parts: {}".format(parts))

first = []
second = []
third = []

for i in range(len(parts)):
    if parts[i] == 0:
        first.append(i)
    elif parts[i] == 1:
        second.append(i)
    elif parts[i] == 2:
        third.append(i)
        
print("first: {}, len: {}".format(first, len(first)))
print("second: {}, len: {}".format(second, len(second)))
print("third: {}, len: {}".format(third, len(third)))


g1 = G.subgraph(first)
g2 = G.subgraph(second)
g3 = G.subgraph(third)

print("g1: {}".format(g1.nodes()))
print("g2: {}".format(g2.nodes()))
print("g3: {}".format(g3.nodes()))

pos = networkx_draw(Ga, nx.spring_layout(Ga))
nx.draw_networkx_edges(Ga, pos, edgelist=g1.edges(), width=3, alpha=0.5, edge_color='r')
plt.savefig('g1.png', dpi=300)

nx.draw_networkx_edges(Ga, pos, edgelist=g2.edges(), width=3, alpha=0.5, edge_color='r')
plt.savefig('g2.png', dpi=300)

nx.draw_networkx_edges(Ga, pos, edgelist=g3.edges(), width=3, alpha=0.5, edge_color='r')
plt.savefig('g3.png', dpi=300)

# union find using networkx
ans_g1 = nx.minimum_spanning_tree(g1)
print("ans_g1: {}".format(ans_g1.nodes()))

pos = networkx_draw(Ga, nx.spring_layout(Ga))
nx.draw_networkx_edges(Ga, pos, edgelist=ans_g1.edges(), width=3, alpha=0.5, edge_color='r')
plt.savefig('graph1.png', dpi=300)

ans_g2 = nx.minimum_spanning_tree(g2)
print("ans_g2: {}".format(ans_g2.nodes()))

pos = networkx_draw(Ga, nx.spring_layout(Ga))
nx.draw_networkx_edges(Ga, pos, edgelist=ans_g2.edges(), width=3, alpha=0.5, edge_color='r')
plt.savefig('graph2.png', dpi=300)

ans_g3 = nx.minimum_spanning_tree(g3)
print("ans_g3: {}".format(ans_g3.nodes()))
pos = networkx_draw(Ga, nx.spring_layout(Ga))
nx.draw_networkx_edges(Ga, pos, edgelist=ans_g3.edges(), width=3, alpha=0.5, edge_color='r')
plt.savefig('graph3.png', dpi=300)