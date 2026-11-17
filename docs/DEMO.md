# Final defense run of show

## Two-minute live demo

Start a fresh app process with `DATABASE_URL= streamlit run app.py`. Open the local URL, choose **Analyze a frame**, and start with the default weights. Restart the process after a rehearsal that saves feedback to restore the starting utility matrix.

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:20 | **Angry face → Analyze frame**. Open **Perception** and **Attention**. | “The CNN estimates a visible expression. Attention accepts this image, and the default recommendation is to continue.” |
| 0:20–0:40 | Select **Analyze frame** again. Open **Memory** and **Reasoning**. | “The window now contains two accepted inputs. The repeated-expression rule raises the smoothed score from about 79% to 84% in this example.” |
| 0:40–1:05 | Open **DSS**. Raise **Expression fit** from 35 to 100. Show the ranking, explanation and sensitivity chart. | “With more weight on fit, a short break moves into first place on the same evidence.” |
| 1:05–1:35 | Reset weights, then raise **Expression fit** to **38**. Show **Decision** and **Action**. In **Learning**, choose **Offer an easier task**, confirm the expression, and save. | “The tutor chooses the action. With the frame and weights fixed, one confirmed choice moves an easier task into first place after the utility update.” |
| 1:35–2:00 | Reset weights. Select **Dark frame → Analyze frame**. Show its rejection and the check-in recommendation in **DSS**. | “The quality gate rejected the frame. The tutor should ask the student directly. Weight changes still update the comparison table.” |

Slide 5 uses a separate simulated batch of five choices per expression. The live path demonstrates a single tutor choice at adjusted weights. Both runs use the same utility-update rule.

## Five-minute presentation

| Time | Slide | Speaking cue |
| --- | --- | --- |
| 0:00–0:30 | 1 | Introduce the team and the tutor’s decision. |
| 0:30–1:00 | 2 | Describe the online lesson, available responses and who chooses. |
| 1:00–1:30 | 3 | Follow one frame through seven stages and trace the feedback loop. |
| 1:30–2:00 | 4 | Explain the face crop, CNN probabilities, frame quality and two thresholds. |
| 2:00–2:30 | 5 | Explain the recent-frame window and how confirmed choices move utility entries. |
| 2:30–3:00 | 6 | Follow the rejected frame into a check-in recommendation and report. |
| 3:00–3:30 | 7 | Explain scoring and why more weight on fit changes the leader. |
| 3:30–4:00 | 8 | Give the exact-deduplicated sample size, accuracy and macro-F1. |
| 4:00–4:30 | 9 | Explain overlapping CK+ subjects, imperfect labels and consent. |
| 4:30–5:00 | 10 | State the outcome and hand over to the live demonstration. |

## Three-minute Q&A preparation

- **Why WSM?** Min-max scaling brings different units onto one scale, and making higher normalized values better handles benefit and cost criteria. Every weighted contribution and the margin over second place are inspectable.
- **Where do the numbers come from?** The starting utility table and disruption estimates are expert assumptions. Time is an estimate in minutes. The default weights prioritize fit. The tutor can change them and see the effect immediately.
- **How does the system handle weak evidence?** Attention checks face detection, quality and confidence. Reasoning uses only accepted history. Rejected frames or an uncertain smoothed expression lead to a check-in. The WSM comparison remains visible.
- **What does learning change?** Confirmed feedback adjusts the fit table by up to 0.05 per affected entry. The app compares the same frame and weights before and after the update. The CNN stays fixed.
- **What do the metrics establish?** They measure expression classification on an image-level test split. Exact matches were removed for the reported subset, but CK+ participants still overlap. The figures do not measure learning outcomes or validate inferred emotions in a classroom.
