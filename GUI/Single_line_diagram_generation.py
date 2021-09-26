"""
Single line diagram drawing for electrical networks
"""

import networkx as nx
import matplotlib.pyplot as plt

class SingleLineDiagramDraming():
    """
    Single line diagram generation
    """
    def __init__(self):
        self.G = nx.Graph()

    def graphic_generation(self,nodes,edges):
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

    def graphic_plot(self, pos, labels):
        # nx.draw(self.G, pos=pos, labels=labels)
        nx.draw_spring(self.G, labels=labels)

        plt.show()


if __name__ == "__main__":
    from pypower import case30
    from pypower.idx_bus import BUS_I
    from pypower.idx_brch import F_BUS,T_BUS
    from pypower import ext2int

    test_graphic = ext2int.ext2int(case30.case30())
    nodes = test_graphic["bus"][:,BUS_I]
    nl = test_graphic["branch"].shape[0]
    nb = test_graphic["bus"].shape[0]

    edges = [(0,0)]*nl
    for i in range(nl):
        edges[i]= (test_graphic["branch"][i,F_BUS],test_graphic["branch"][i,T_BUS])
    pos = dict()
    labels = dict()
    for i in range(nb):
        pos[test_graphic["bus"][i,BUS_I]] = [i*10, i*10+5]
        labels[test_graphic["bus"][i,BUS_I]] = "bus{0}".format(i+1)

    single_line_diagram_draming = SingleLineDiagramDraming()
    single_line_diagram_draming.graphic_generation(nodes=nodes,edges=edges)
    single_line_diagram_draming.graphic_plot(pos=pos,labels=labels)
