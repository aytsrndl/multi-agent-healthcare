from src.topologies.fully_connected import FullyConnectedTopology


class DecentralizedTopology(FullyConnectedTopology):
    """
    Paper-aligned decentralized multi-agent topology.

    Protocol:
        1. All clinical agents independently generate revisions.
        2. Every agent receives the complete set of peer proposals.
        3. Every agent may retain, refine, or replace its proposal.
        4. Final candidates are deduplicated and adjudicated using
           the shared candidate-selection mechanism.

    This corresponds to all-to-all peer communication with no
    central orchestrator.

    Adapted from:
    Kim et al., "Towards a Science of Scaling Agent Systems",
    MAS-Decentralized architecture.
    """

    pass