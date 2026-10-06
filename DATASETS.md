# Dataset sources

Datasets are downloaded by the Docker Compose `dataset-downloader` service. They are not redistributed in this project.

| Dataset | Kaggle handle | Notes |
|---|---|---|
| FER-2013 | `msambare/fer2013` | Seven standard expression classes; supports folder images and FER pixel CSV parsing. |
| CK+ / CK+48 | `shawon10/ckplus` | Lab expressions; contempt is omitted from the seven-class target. |
| RAF-DB | `shuvoalok/raf-db-dataset` | In-the-wild expressions; source layouts may require parser adjustments. |

Target labels: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. The normalizer recognizes common textual labels, FER-2013 integer labels and RAF-DB's 1-7 folder convention. A deterministic 80/10/10 split uses seed 42 by default; image cap is configurable with `MAX_PER_CLASS_SOURCE`.

Run:

```bash
cp .env.example .env
docker compose --profile data run --rm dataset-downloader
```

Kaggle access, terms of service, and licenses vary by dataset. Review and accept them as required. Never commit Kaggle credentials or distribute downloaded images without permission. The application persists prediction metadata only, not raw user images.
