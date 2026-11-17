"""Before/after demo of the Learning module (Role B).

Runs the same frame sequence twice: first with the original utility matrix M, then
after simulated tutor feedback has updated M. The tutor in this script is scripted:
for each emotion it rejects the system's recommendation and picks the 2nd-ranked action.

Usage:
    python scripts/demo_learning.py             # in-memory storage, no database needed
    python scripts/demo_learning.py --use-db    # also stores feedback/versions in PostgreSQL
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emotion_system import action, decision, reasoning  # noqa: E402
from src.emotion_system.config import ALTERNATIVES, DATABASE_URL, EMOTIONS, UTILITY  # noqa: E402
from src.emotion_system.learning import apply_feedback  # noqa: E402
from src.emotion_system.memory import Memory  # noqa: E402
from src.emotion_system.state import CogState  # noqa: E402

FEEDBACK_PER_EMOTION = 5
UNREACHABLE_DB = "postgresql://nobody:nobody@localhost:1/none"  # forces in-memory fallback


def peaked(emotion, p=0.9):
    rest = (1 - p) / (len(EMOTIONS) - 1)
    return {e: p if e == emotion else rest for e in EMOTIONS}


def run_frame(memory, emotion, session_id):
    """Run one frame through Reasoning -> Decision -> Action and return the CogState."""
    probs = peaked(emotion)
    memory.sessions[session_id].append(
        {"accepted": True, "probabilities": probs, "emotion": emotion})
    state = CogState(session_id=session_id, emotion=emotion, probs=probs, confidence=0.9,
                     quality=0.8, face_found=True, accepted=True)
    for stage in (reasoning, decision, action):
        state = stage.run(state, memory)
    return state


def summarize(state):
    return {"recommended": state.recommended,
            "ranking": [(r["alternative"], round(r["score"], 3)) for r in state.ranking],
            "decision_id": state.report["decision_id"]}

def rank_of(result, alternative):
    return [name for name, _ in result["ranking"]].index(alternative) + 1

def score_of(result, alternative):
    return dict(result["ranking"])[alternative]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-db", action="store_true", help="store results in PostgreSQL")
    args = parser.parse_args()

    memory = Memory(DATABASE_URL if args.use_db else UNREACHABLE_DB)
    memory.utility = copy.deepcopy(UTILITY)  # always start the demo from the original matrix
    print(f"Storage: {'PostgreSQL' if memory.db_ok else 'in-memory only'}\n")

    # ---- Run 1: original matrix M ----
    before = {e: summarize(run_frame(memory, e, f"demo-before-{e}")) for e in EMOTIONS}

    # ---- Simulated tutor feedback ----
    for emotion, result in before.items():
        recommended = result["recommended"]
        chosen = result["ranking"][1][0]  # tutor prefers the system's 2nd-ranked action
        for _ in range(FEEDBACK_PER_EMOTION):
            memory.save_feedback(f"demo-before-{emotion}", emotion, recommended, chosen,
                                 True, decision_id=result["decision_id"])
        result["tutor_choice"] = chosen

    # ---- Learning ----
    summary = apply_feedback(memory)
    print("Learning:", summary, "\n")

    # ---- Run 2: same frames, updated matrix M ----
    after = {e: summarize(run_frame(memory, e, f"demo-after-{e}")) for e in EMOTIONS}

    # ---- Report ----
    print(f"{'emotion':<10}{'before':<14}{'tutor wanted':<14}{'after':<14}{'changed?':<10}{'rank':<10}score gap to #1")
    changed = 0
    for emotion in EMOTIONS:
        b, a = before[emotion], after[emotion]
        flag = "YES" if a["recommended"] != b["recommended"] else "no"
        changed += flag == "YES"
        choice = b["tutor_choice"]
        gap_before = b["ranking"][0][1] - score_of(b, choice)
        gap_after = a["ranking"][0][1] - score_of(a, choice)
        rank_txt = f"#{rank_of(b, choice)}->#{rank_of(a, choice)}"
        print(f"{emotion:<10}{b['recommended']:<14}{choice:<14}{a['recommended']:<14}{flag:<10}{rank_txt:<10}{gap_before:.3f} -> {gap_after:.3f}")
    print(f"\nRecommendation changed for {changed} of {len(EMOTIONS)} emotions.")

    print("\nMatrix M, row for 'angry' (order: " + ", ".join(ALTERNATIVES) + ")")
    print("  before:", UTILITY["angry"])
    print("  after: ", memory.utility["angry"])

    out = ROOT / "output" / "learning_demo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "simulated_tutor": True, "feedback_per_emotion": FEEDBACK_PER_EMOTION,
        "learning_summary": summary, "matrix_before": UTILITY, "matrix_after": memory.utility,
        "before": before, "after": after}, indent=2), encoding="utf-8")
    print(f"\nSaved comparison to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()