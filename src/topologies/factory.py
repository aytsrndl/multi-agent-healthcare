from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.independent import IndependentTopology
from src.topologies.star import StarTopology
from src.topologies.tree import TreeTopology


def create_topology(topology_name: str, agents):
    name = topology_name.strip().lower()

    if name == "fully_connected":
        return FullyConnectedTopology(agents)

    if name == "independent":
        return IndependentTopology(agents)

    if name == "star":
        return StarTopology(agents)

    if name == "tree":
        return TreeTopology(agents)
    
    raise ValueError(
        f"Unsupported topology: {topology_name}"
    )