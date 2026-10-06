# Upload examples

Six individual face crops from the CC0 image **Diverse Facial Expressions**.

- Source: https://www.publicdomainpictures.net/en/view-image.php?image=773242&picture=diverse-facial-expressions
- License: CC0; the source page says attribution is not required.
- Downloaded standard image: `cc0-diverse-facial-expressions.jpg` (1920 × 1920).
- The source image is marked AI Generated. These are synthetic test faces, not photographs of real people.

Open the app at `http://localhost:8501`, select **Upload image**, and choose any `*-face.jpg` file. Verified with the current model and app thresholds (`quality ≥ 0.45`, `confidence ≥ 0.25`, and face detected):

| File | Expected visual expression | Model output | Confidence | Quality | Passed |
| --- | --- | --- | ---: | ---: | --- |
| `laughing-face.jpg` | Laughing / joyful | angry | 58.8% | 0.983 | Yes |
| `fear-face.jpg` | Fearful | surprise | 41.8% | 0.971 | Yes |
| `smiling-face.jpg` | Happy | happy | 95.8% | 0.908 | Yes |
| `sad-face.jpg` | Sad | sad | 29.2% | 0.865 | Yes |
| `angry-face.jpg` | Angry | angry | 70.0% | 0.992 | Yes |
| `surprised-face.jpg` | Surprised | happy | 60.8% | 0.912 | Yes |

Expected visual expressions describe the artwork, not verified ground-truth labels. The differences are useful for demonstrating that passing Attention means the image is usable; it does not guarantee the emotion prediction is correct.
