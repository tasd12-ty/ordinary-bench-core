# Test Data — Quick Start

This directory contains **140 test scenes** (20 per complexity level, n04-n10) with pre-generated images and questions, ready for immediate VLM evaluation.

## Contents

```
test-data/
├── scenes/              # 140 scene JSON files (3D object positions, properties)
├── images/
│   ├── single_view/     # 140 PNG images (480x320, one per scene)
│   └── multi_view/      # 140 directories, each with view_0..view_3.png
├── questions/
│   ├── qrr/             # QRR questions (disjoint + shared_anchor variants)
│   ├── trr/             # TRR questions (clock-face direction)
│   └── fdr/             # FDR questions (full distance ranking)
└── subset_ablation/     # 912 subset scenes for object-count sensitivity testing
    ├── scenes/          # Subset scene JSONs
    ├── images/single_view/  # Subset images
    ├── master_questions/    # Master QRR question banks (10 parent scenes)
    ├── manifest.json        # Subset enumeration manifest
    └── generate_questions.py  # Run to generate questions locally
```

## Quick Start — Run VLM Evaluation

### 1. Set up API credentials

```bash
export VLM_API_KEY="your-api-key"
export VLM_BASE_URL="https://openrouter.ai/api/v1"   # or your endpoint
```

### 2. Run evaluation with a job TOML

```bash
cd VLM-test/API-test

# Edit the job TOML to set your model
python run_eval.py --job jobs/example.toml
```

Or create a minimal job TOML pointing to test-data:

```toml
job_name = "test_run"

[provider]
adapter = "openai_chat"
model = "openai/gpt-4o"
base_url = "https://openrouter.ai/api/v1"
api_key = "env:VLM_API_KEY"

[provider.options]
temperature = 0.0
max_tokens = 65536
max_concurrency = 4

[input]
questions_dir = "../../datasets/test-data/questions"
question_layout = "v2"
question_types = ["qrr", "trr", "fdr"]
batch_size = 20

[images]
mode = "single"
single_view_root = "../../datasets/test-data/images/single_view"

[output]
results_dir = "./results"
run_name = "test_run"
```

### 3. Check results

```
results/test_run/
├── raw/            # Raw VLM responses
├── scenes/         # Per-scene scores
└── summary.json    # Aggregated accuracy
```

## Subset Ablation

The subset data requires one extra step — generate questions locally (deterministic, ~5 seconds):

```bash
cd datasets/test-data/subset_ablation
python generate_questions.py
# Creates questions/qrr/ with 912 files (624,963 questions)
```

## Data Verification

```bash
# Check scene count
ls scenes/ | wc -l                    # Expected: 140

# Check QRR has both variants
python3 -c "
import json, glob
d = s = 0
for f in glob.glob('questions/qrr/*.json'):
    data = json.load(open(f))
    for b in data['batches']:
        for q in b['questions']:
            if q['variant'] == 'disjoint': d += 1
            elif q['variant'] == 'shared_anchor': s += 1
print(f'disjoint: {d}, shared_anchor: {s}')
"
# Expected: disjoint: 26136, shared_anchor: 18583

# Check images exist for all scenes
ls images/single_view/ | wc -l       # Expected: 140
ls images/multi_view/ | wc -l        # Expected: 140
```

## Scene ID Convention

Scene IDs follow the pattern `{split}_{index:06d}`:
- `n04_000080` — 4 objects, index 80 (test split starts at 80)
- `n10_000099` — 10 objects, index 99

Splits: `n04` through `n10` (4 to 10 objects per scene).
