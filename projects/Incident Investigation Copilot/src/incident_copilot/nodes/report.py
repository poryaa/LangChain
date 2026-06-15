from ..state import IncidentState


def generate_report(state: IncidentState) -> IncidentState:
    """
    Phase 0: identity node.
    Later you can format markdown/HTML or add sections here.
    """
    # final_answer is already prepared in synthesize_evidence
    return state