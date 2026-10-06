class Attention:
    def __init__(self, quality_threshold=0.45, confidence_threshold=0.55):
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold

    def evaluate(self, perception):
        quality = perception["quality_score"]
        confidence = perception["confidence"]
        relevance = 0.45 * quality + 0.55 * confidence
        reasons = []
        if quality < self.quality_threshold:
            reasons.append("quality below threshold")
        if confidence < self.confidence_threshold:
            reasons.append("confidence below threshold")
        accepted = not reasons
        return {
            "accepted": accepted,
            "quality_threshold": self.quality_threshold,
            "confidence_threshold": self.confidence_threshold,
            "relevance": float(relevance),
            "reason": "accepted" if accepted else "; ".join(reasons),
        }
