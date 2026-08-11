import copy

from src.agents.agent_manager import Agent, CoTAgent, IOAgent, RandomAgent

AGENT_REGISTRY = {
    "IOAgent": IOAgent,
    "CoTAgent": CoTAgent,
    "RandomAgent": RandomAgent,
}


def create_agent(agent_config: dict) -> Agent:
    """Create an agent instance from config provided."""
    agent_class = AGENT_REGISTRY.get(agent_config["type"])
    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    agent = agent_class(agent_config=agent_config)
    return agent


def create_players_with_player_id(
    agent_cfgs: list[dict], num_players: int
) -> list[Agent]:
    """Create players from agent configurations.

    If an agent config already has a ``player_id`` key, that agent is pinned to
    that single position (useful for one-sided experiments, e.g. Gemini=P1 vs
    Random=P2 only).  If ``player_id`` is absent, the agent is assigned to every
    position 1..num_players as before (round-robin tournament mode).
    """
    players = []
    for cfg in agent_cfgs:
        if "player_id" in cfg:
            agent_config = copy.deepcopy(cfg)
            agent = create_agent(agent_config)
            players.append(agent)
        else:
            for player_id in range(1, num_players + 1):
                agent_config = copy.deepcopy(cfg)
                agent_config["player_id"] = player_id
                agent = create_agent(agent_config)
                players.append(agent)
    return players
