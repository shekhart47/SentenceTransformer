import math

def count_lines(file_path):
    # Skip header line
    with open(file_path, 'r') as f:
        return sum(1 for _ in f) - 1

train_samples = count_lines(config.TRAIN_DATASET_PATH)
per_device = config.TRAIN_BATCH_SIZE  # e.g. 512
world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
batches_per_epoch = train_samples // (per_device * world_size)
optimizer_steps_per_epoch = batches_per_epoch // config.GRADIENT_ACCUMULATION_STEPS
max_steps = optimizer_steps_per_epoch * config.EPOCHS

args = SentenceTransformerTrainingArguments(
    ...,
    max_steps=max_steps,        # required when len(train_dataset) is unknown
    num_train_epochs=None,      # don't specify epochs when using max_steps
    warmup_ratio=0.1,           # use a ratio instead of warmup_steps
    ...
)
