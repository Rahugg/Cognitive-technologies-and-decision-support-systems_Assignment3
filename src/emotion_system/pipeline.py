from . import action, decision, learning, reasoning
from .attention import Attention
from .config import CONFIDENCE_THRESHOLD
from .knowledge import Knowledge
from .memory import Memory
from .perception import Perception
from .state import CogState


class CognitiveEmotionSystem:
    def __init__(self, model, emotions, database_url, quality_threshold=0.45,
                 confidence_threshold=CONFIDENCE_THRESHOLD, memory_window=5, device="cpu"):
        self.perception = Perception(model, emotions, device)
        self.attention = Attention(quality_threshold, confidence_threshold)
        self.memory = Memory(database_url, memory_window)
        self.knowledge = Knowledge()  # graph kept for reference; rules now run in Reasoning

    def run_cycle(self, image, session_id, require_face=False) -> CogState:
        """Perception -> Attention -> Memory -> Reasoning -> Decision -> Action -> Learning."""
        state = CogState(session_id=session_id, image=image)
        perception = self.perception.process(image)
        state.emotion, state.probs = perception["emotion"], perception["probabilities"]
        state.confidence, state.quality = perception["confidence"], perception["quality_score"]
        state.face_found = perception["face_found"]
        state.brightness, state.sharpness = perception["brightness"], perception["sharpness"]
        state.explanation.append(
            f"Perception: {state.emotion} ({state.confidence:.0%}), quality {state.quality:.2f}, "
            f"face {'found' if state.face_found else 'not found'}.")

        attention = self.attention.evaluate(perception)
        state.accepted, state.relevance = attention["accepted"], attention["relevance"]
        state.reject_reason = "" if state.accepted else attention["reason"]
        if require_face and not state.face_found:
            state.accepted = False
            state.reject_reason = "; ".join(filter(None, ["face not detected", state.reject_reason]))
        state.explanation.append(
            "Attention: accepted." if state.accepted else f"Attention: rejected ({state.reject_reason}).")

        previous = self.memory.previous(session_id)
        state.previous_emotion = previous["emotion"] if previous else None
        attention["accepted"] = state.accepted
        self.memory.record(session_id, perception, attention, None)
        state.explanation.append(
            f"Memory: previous accepted emotion {state.previous_emotion or 'none'}; "
            f"{len(self.memory.recent(session_id))} frame(s) in short-term window.")

        for stage in (reasoning, decision, action, learning):
            state = stage.run(state, self.memory)
        state.image = None  # raw pixels are never kept past the cycle
        return state

    def process(self, image, session_id, require_face=False):
        """Backward-compatible dict for app.py and scripts/run_demo.py."""
        state = self.run_cycle(image, session_id, require_face)
        smoothed = {"emotion": max(state.smoothed_probs, key=state.smoothed_probs.get)
                    if state.smoothed_probs else state.emotion,
                    "confidence": state.smoothed_confidence,
                    "probabilities": state.smoothed_probs}
        return {
            "perception": {"emotion": state.emotion, "confidence": state.confidence,
                           "probabilities": state.probs, "brightness": state.brightness,
                           "sharpness": state.sharpness, "quality_score": state.quality,
                           "face_found": state.face_found},
            "attention": {"accepted": state.accepted, "quality_threshold": self.attention.quality_threshold,
                          "confidence_threshold": self.attention.confidence_threshold,
                          "relevance": state.relevance,
                          "reason": "accepted" if state.accepted else state.reject_reason},
            "memory": {"previous": {"emotion": state.previous_emotion} if state.previous_emotion else None,
                       "current": None, "smoothed_prediction": smoothed,
                       "short_term_size": len(self.memory.recent(session_id))},
            "knowledge": {"label": state.label, "action": state.recommended,
                          "confidence": state.smoothed_confidence, "rules": state.rules},
            "state": state,
        }
