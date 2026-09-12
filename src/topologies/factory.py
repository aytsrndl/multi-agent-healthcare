from src.topologies.fully_connected import FullyConnectedTopology


def create_topology(topology_name: str, agents):
    name = topology_name.strip().lower()

    if name == "fully_connected":
        return FullyConnectedTopology(agents)

    raise ValueError(f"Unsupported topology: {topology_name}")