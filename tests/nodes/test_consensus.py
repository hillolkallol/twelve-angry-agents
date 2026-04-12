from twelve_angry_agents.nodes.consensus import check_consensus, is_hung_jury, majority_vote


def test_unanimous_votes_return_true():
    votes = {
        "The Analyst": "proceed",
        "The Skeptic": "proceed",
        "The Optimist": "proceed",
    }
    assert check_consensus(votes) is True


def test_split_votes_return_false():
    votes = {
        "The Analyst": "proceed",
        "The Skeptic": "don't proceed",
        "The Optimist": "proceed",
    }
    assert check_consensus(votes) is False


def test_single_dissenter_not_unanimous():
    votes = {f"Agent{i}": "proceed" for i in range(11)}
    votes["The Contrarian"] = "don't proceed"
    assert check_consensus(votes) is False


def test_all_same_option_is_unanimous():
    votes = {f"Agent{i}": "don't proceed" for i in range(12)}
    assert check_consensus(votes) is True


def test_is_hung_jury_when_max_rounds_reached():
    assert is_hung_jury(current_round=50, max_rounds=50) is True


def test_is_not_hung_jury_before_max_rounds():
    assert is_hung_jury(current_round=3, max_rounds=50) is False


def test_majority_vote_calculation():
    votes = {
        "Agent1": "proceed",
        "Agent2": "proceed",
        "Agent3": "proceed",
        "Agent4": "don't proceed",
        "Agent5": "don't proceed",
    }
    result = majority_vote(votes)
    assert result == "proceed"


def test_majority_vote_returns_tied_option():
    votes = {
        "Agent1": "proceed",
        "Agent2": "don't proceed",
    }
    # Tie — either option acceptable, just must be one of the two
    result = majority_vote(votes)
    assert result in ("proceed", "don't proceed")
