import networkx as nx

def create_network(net_type: str, n: int, seed: int | None = None, **params ):
    """
    Create a network of a given type.

    net_type:
        "BA" = Barabasi-Albert
        "ER" = Erdos-Renyi
        "WS" = Watts-Strogatz

    Common:
        n = number of nodes

    BA params:
        m = edges per new node

    ER params:
        p = connection probability

    WS params:
        k = nearest neighbors
        p = rewiring probability
    """

    net_type = net_type.upper()

    if net_type == "BA":
        m = params.get("m", 2)
        return nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    elif net_type == "ER":
        p = params.get("p", 0.1)
        return nx.erdos_renyi_graph(n=n, p=p, seed=seed)

    elif net_type == "WS":
        k = params.get("k", 4)
        p = params.get("p", 0.1)
        return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

    else:
        raise ValueError(f"Unknown network type: {net_type}")
