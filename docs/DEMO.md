# Two-minute live demo

Before the defense, run `DATABASE_URL= streamlit run app.py`, open the local URL, and use a fresh browser session. Use the built-in frames. Keep the DSS weights at their defaults and use the in-memory storage mode to reproduce the starting utility matrix. Restart the server after rehearsing feedback to restore the matrix.

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:20 | **Angry face → Analyze frame** | “The CNN finds a face and estimates the visible expression. At the default weights, the DSS ranks continuing first.” |
| 0:20–0:40 | **Perception**, **Attention**, **Memory**, **Reasoning** | “Here are the probabilities and quality gate. Reasoning gives recent accepted frames more weight.” |
| 0:40–1:05 | Open **DSS**, raise **Expression fit** from 35 to 100, then reset. | “A short break moves into first place on the same frame. Here’s the runner-up comparison and the sensitivity chart.” |
| 1:05–1:35 | Show **Decision**, **Action**, and **Learning**. Choose **Offer an easier task**, confirm the predicted expression looked accurate, and save. | “Action generates a report. My tutoring choice updates the utility matrix. The before/after table shows the effect on fit and scores.” |
| 1:35–2:00 | **Dark frame → Analyze frame**, then open **Attention**. | “Attention rejects this low-quality frame. There is no accepted history here, so the default recommendation is to ask the student directly. The tutor makes the final decision.” |

For the five-minute presentation, use the ten slides in order and budget about 30 seconds per slide. Explain the test metrics separately from the scripted tutor feedback. The model classifies visible expressions. The student can clarify what they feel, and the tutor decides how to respond.
