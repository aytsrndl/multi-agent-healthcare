from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.independent import IndependentTopology
from src.topologies.star import StarTopology
from src.topologies.tree import TreeTopology
from src.topologies.decentralized import DecentralizedTopology
from src.topologies.centralized import CentralizedTopology
from src.topologies.hybrid import HybridTopology


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

    if name == "decentralized":
        return DecentralizedTopology(agents)

    if name == "centralized":
        return CentralizedTopology(agents)

    if name == "hybrid":
        return HybridTopology(agents)
    
    raise ValueError(
        f"Unsupported topology: {topology_name}"
    )
