from config.roles import (
    INFANT_CARE_ROLES,
    PHYSICAL_RECOVERY_ROLES,
    MENTAL_HEALTH_ROLES,
)

from src.agent import Agent
from src.schemas import Domain
from src.llm.base import LLMClient


ROLE_MAP = {
    Domain.INFANT_CARE: INFANT_CARE_ROLES,
    Domain.PHYSICAL_RECOVERY: PHYSICAL_RECOVERY_ROLES,
    Domain.MENTAL_HEALTH: MENTAL_HEALTH_ROLES,
}


def get_roles_for_domain(
    domain: Domain,
) -> dict[str, str]:
    """
    Return the configured healthcare provider roles
    for a medical domain.
    """

    if domain not in ROLE_MAP:
        raise ValueError(
            f"No healthcare roles configured for domain: {domain}"
        )

    return ROLE_MAP[domain]


def create_agents_for_domain(
    domain: Domain,
    llm: LLMClient,
) -> list[Agent]:

    roles = get_roles_for_domain(domain)

    agents = [
        Agent(
            name=name,
            role_description=description,
            llm=llm,
        )
        for name, description in roles.items()
    ]

    return agents