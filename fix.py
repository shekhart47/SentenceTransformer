from transformers import AutoTokenizer, DataCollatorWithPadding

class TripletCollator:
    """
    Custom collator for anchor, positive, negative sequences.  It defines
    a __call__ method for batching and exposes valid_label_columns (empty).
    """
    def __init__(self, model_name, pad_to_multiple=8):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # DataCollatorWithPadding handles dynamic padding and device tensors
        self.base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=pad_to_multiple,
            return_tensors='pt'
        )
        # SentenceTransformerTrainer looks at this attribute to detect
        # which dataset columns are labels.  We have none.
        self.valid_label_columns = []

    def __call__(self, features):
        # Extract anchor, positive and negative lists from the batch
        anchors   = [example['anchor']    for example in features]
        positives = [example['positives'] for example in features]
        negatives = [example['negatives'] for example in features]

        # Tokenize and pad each field separately
        anchor_batch   = self.base_collator.tokenizer(
            anchors, padding=True, truncation=True, return_tensors='pt'
        )
        positive_batch = self.base_collator.tokenizer(
            positives, padding=True, truncation=True, return_tensors='pt'
        )
        negative_batch = self.base_collator.tokenizer(
            negatives, padding=True, truncation=True, return_tensors='pt'
        )

        return {
            'anchor':    anchor_batch,
            'positives': positive_batch,
            'negatives': negative_batch
        }



data_collator = TripletCollator(config.MODEL_NAME, pad_to_multiple=8)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
    data_collator=data_collator,
    callbacks=[...],
    optimizers=(optimizer, scheduler),
)
