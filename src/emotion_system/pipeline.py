from .attention import Attention
from .knowledge import Knowledge
from .memory import Memory
from .perception import Perception


class CognitiveEmotionSystem:
    def __init__(self, model, emotions, database_url, quality_threshold=0.45,
                 confidence_threshold=0.25, memory_window=5, device="cpu"):
        self.perception = Perception(model, emotions, device)
        self.attention = Attention(quality_threshold, confidence_threshold)
        self.memory = Memory(database_url, memory_window)
        self.knowledge = Knowledge()

    def process(self, image, session_id, require_face=False):
        perception = self.perception.process(image)
        attention = self.attention.evaluate(perception)
        if require_face and not perception["face_found"]:
            attention["accepted"] = False
            attention["reason"] = (
                "face not detected" if attention["reason"] == "accepted"
                else f"face not detected; {attention['reason']}"
            )
        previous = self.memory.previous(session_id)
        smoothed = self.memory.smooth_prediction(session_id, perception, attention["accepted"])
        decision_perception = {**perception, **smoothed}
        decision = self.knowledge.infer(decision_perception, attention, previous)
        memory_record = self.memory.record(session_id, perception, attention, decision)
        return {
            "perception": {k: v for k, v in perception.items() if k != "face_crop"},
            "attention": attention,
            "memory": {"previous": previous, "current": memory_record,
                       "smoothed_prediction": smoothed,
                       "short_term_size": len(self.memory.recent(session_id))},
            "knowledge": decision,
        }
