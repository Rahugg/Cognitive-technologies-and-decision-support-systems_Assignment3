# Demo images

The face crops come from the CC0 illustration **Diverse Facial Expressions** on [Public Domain Pictures](https://www.publicdomainpictures.net/en/view-image.php?image=773242&picture=diverse-facial-expressions). The crops retain that license. These illustrations stand in for lesson frames. Their visual labels describe the artwork.

`dark-smiling-face.jpg` uses the same crop as `smiling-face.jpg` with brightness reduced to 20% and a 2 px Gaussian blur. It exercises the low-quality path with an otherwise recognizable face.

Verified with the bundled checkpoint, default gate thresholds and a fresh analysis window per image:

| Image | CNN expression | Confidence | Quality | Attention | Default recommendation |
| --- | --- | ---: | ---: | --- | --- |
| Angry face | angry | 79.0% | 0.992 | Accepted | Continue the lesson |
| Dark frame | happy | 86.1% | 0.367 | Rejected | Ask the student directly |
| Fearful face | surprise | 41.4% | 0.971 | Rejected | Ask the student directly |
| Laughing face | angry | 64.8% | 0.983 | Accepted | Continue the lesson |
| Sad face | sad | 36.6% | 0.865 | Rejected | Ask the student directly |
| Clear smiling face | happy | 96.1% | 0.908 | Accepted | Continue the lesson |
| Surprised face | happy | 61.1% | 0.912 | Accepted | Continue the lesson |

The default gate requires a detected face, quality of at least 0.45 and confidence of at least 0.55. The angry example ranks **Take a short break** first when only the expression fit slider is raised from 35 to 100. The surprised example passes Attention even though the model's label differs from the artwork label.
