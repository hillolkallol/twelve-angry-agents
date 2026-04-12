from collections import Counter


def check_consensus(votes: dict[str, str]) -> bool:
    """Return True if all agents agree on the same vote option."""
    if not votes:
        return False
    return len(set(votes.values())) == 1


def is_hung_jury(current_round: int, max_rounds: int) -> bool:
    """Return True if max deliberation rounds have been reached."""
    return current_round >= max_rounds


def majority_vote(votes: dict[str, str]) -> str:
    """Return the most common vote option. Breaks ties by picking the first."""
    counter = Counter(votes.values())
    return counter.most_common(1)[0][0]
