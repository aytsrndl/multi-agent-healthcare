from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.independent import IndependentTopology


def create_topology(topology_name: str, agents):
    name = topology_name.strip().lower()

    if name == "fully_connected":
        return FullyConnectedTopology(agents)

    if name == "independent":
        return IndependentTopology(agents)

    raise ValueError(
        f"Unsupported topology: {topology_name}"
    )