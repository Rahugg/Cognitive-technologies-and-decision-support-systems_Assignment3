import networkx as nx


class Knowledge:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([
            ("Emotion", {"type": "concept"}),
            ("Condition", {"type": "concept"}),
            ("Action", {"type": "concept"}),
            ("MemoryFact", {"type": "concept"}),
            ("request_more_data", {"type": "action"}),
            ("uncertain", {"type": "action"}),
        ])
        self.graph.add_edge("Emotion", "Action", relation="maps_to")
        self.graph.add_edge("Condition", "request_more_data", relation="triggers")
        self.graph.add_edge("MemoryFact", "Emotion", relation="supports")
        self.actions = {
            "angry": "calm_response", "disgust": "neutral_response", "fear": "reassurance",
            "happy": "positive_feedback", "neutral": "neutral_response", "sad": "supportive_response",
            "surprise": "acknowledge_surprise",
        }

    def infer(self, perception, attention, previous):
        rules = []
        if not attention["accepted"]:
            rules.append("R1: low relevance or weak quality -> request_more_data")
            return {"label": "insufficient_data", "action": "request_more_data",
                    "confidence": perception["confidence"], "rules": rules}

        confidence = perception["confidence"]
        if previous and previous["emotion"] == perception["emotion"]:
            confidence = min(1.0, confidence + 0.05)
            rules.append("R2: previous accepted emotion matches -> continuity bonus +0.05")
        if confidence < 0.60:
            rules.append("R3: adjusted confidence below 0.60 -> uncertain")
            return {"label": "uncertain", "action": "request_more_data",
                    "confidence": confidence, "rules": rules}
        action = self.actions[perception["emotion"]]
        rules.append(f"R4: accepted {perception['emotion']} -> {action}")
        return {"label": perception["emotion"], "action": action,
                "confidence": confidence, "rules": rules}
