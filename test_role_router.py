from src.role_router import create_agents_for_domain
from src.schemas import Domain


domains = [
    Domain.INFANT_CARE,
    Domain.PHYSICAL_RECOVERY,
    Domain.MENTAL_HEALTH,
]


for domain in domains:

    print("\n")
    print("=" * 70)
    print(domain.value)
    print("=" * 70)

    agents = create_agents_for_domain(domain)

    print(f"Number of agents: {len(agents)}")

    for agent in agents:
        print(f"- {agent.name}")