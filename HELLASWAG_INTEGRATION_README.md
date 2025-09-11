# HellaSwag Evaluation Integration

This document describes the integration of HellaSwag evaluation into the pretraining process.

## Overview

HellaSwag evaluation has been integrated into the pretraining pipeline to provide additional evaluation metrics during training. The evaluation runs during validation steps and measures the model's ability to predict plausible continuations for given contexts.

## Features

- **Automatic HellaSwag evaluation** during validation steps
- **Support for both sync and async validation modes**
- **Configurable batch size and sample limits** for evaluation
- **Results logging** to TensorBoard and local files
- **Standalone evaluation script** for independent testing

## Usage

### During Pretraining

Add the following arguments to your pretraining command:

```bash
python accessory/main_pretrain.py \
    --hellaswag_eval \
    --hellaswag_data_dir /path/to/hellaswag/data \
    --hellaswag_batch_size 4 \
    --hellaswag_max_samples 1000 \
    [other pretraining arguments...]
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--hellaswag_eval` | Enable HellaSwag evaluation during validation | False |
| `--hellaswag_data_dir` | Directory containing HellaSwag data | `data/hellaswag/` |
| `--hellaswag_batch_size` | Batch size for HellaSwag evaluation | 4 |
| `--hellaswag_max_samples` | Max number of samples to evaluate (None for all) | None |

### Data Setup

The integration expects HellaSwag validation data in JSONL format at `{hellaswag_data_dir}/hellaswag_val.jsonl`.

If the data file doesn't exist, the system will attempt to automatically download it using the HuggingFace `datasets` library.

To manually prepare the data:

```python
from datasets import load_dataset
import jsonlines

# Download HellaSwag validation data
dataset = load_dataset("hellaswag", split="validation")

# Save to local file
with jsonlines.open("data/hellaswag/hellaswag_val.jsonl", mode='w') as writer:
    for item in dataset:
        writer.write(item)
```

## Standalone Evaluation

You can also run HellaSwag evaluation independently using the standalone script:

```bash
python light-eval/src/eval_hellaswag.py \
    --pretrained_path /path/to/model \
    --data_dir /path/to/hellaswag/data \
    --batch_size 8
```

## Output

### Console Output

During pretraining, you'll see output like:

```
[HellaSwag Eval - Iter 10000] Accuracy: 0.3245 (324/1000)
```

### File Output

Results are saved to:
- `{output_dir}/hellaswag_results/iter_{iteration}.json` - Individual iteration results
- `{output_dir}/hellaswag_results/hellaswag_log.jsonl` - Continuous log of all evaluations

### TensorBoard

Metrics are logged to TensorBoard under the `hellaswag/` namespace:
- `hellaswag/accuracy`
- `hellaswag/total_samples`
- `hellaswag/correct_samples`

## Implementation Details

### Evaluation Method

The evaluation uses a perplexity-based approach:

1. For each HellaSwag example, calculate the perplexity of the full text (context + ending) for each possible ending
2. Select the ending with the lowest perplexity as the prediction
3. Compare with the ground truth label to determine correctness

### Integration Points

The evaluation is integrated at two points in the training loop:

1. **Sync validation**: Runs on the main process during regular validation steps
2. **Async validation**: Runs in the background validation process when `--async_val` is enabled

### Performance Considerations

- HellaSwag evaluation adds computational overhead to validation
- Use `--hellaswag_max_samples` to limit evaluation size for faster iterations
- Consider using `--async_val` to avoid blocking training
- Adjust `--hellaswag_batch_size` based on available GPU memory

## Files Modified/Added

### New Files
- `accessory/util/hellaswag_eval.py` - Core evaluation utilities
- `light-eval/src/eval_hellaswag.py` - Standalone evaluation script

### Modified Files
- `accessory/engine_pretrain.py` - Added evaluation calls to training loop
- `accessory/main_pretrain.py` - Added command line arguments

## Error Handling

The integration includes robust error handling:
- Missing data files trigger automatic download attempts
- Evaluation failures are logged as warnings but don't stop training
- Invalid samples are handled gracefully

## Example Integration

Here's a complete example of running pretraining with HellaSwag evaluation:

```bash
python accessory/main_pretrain.py \
    --llama_type llama \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_meta_path /path/to/training/data \
    --output_dir ./output \
    --batch_size 4 \
    --val_freq 5000 \
    --hellaswag_eval \
    --hellaswag_data_dir ./data/hellaswag \
    --hellaswag_batch_size 4 \
    --hellaswag_max_samples 500
```

This will run HellaSwag evaluation every 5000 iterations on 500 samples with a batch size of 4.