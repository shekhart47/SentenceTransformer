import torch
from transformers import AutoTokenizer

class TripletCollator:
    """
    Collator for (anchor, positives, negatives) triplets expected by
    SentenceTransformerTrainer.  It pads each field separately and returns
    a list of dicts: [anchor_batch, positive_batch, negative_batch].
    """
    def __init__(self, model_name: str, pad_to_multiple_of: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_to_multiple_of = pad_to_multiple_of
        # no label columns for triplet/MNR training
        self.valid_label_columns = []

    def _pad_batch(self, tokenized_examples):
        """
        Given a dict with lists of 'input_ids' and 'attention_mask', pad them
        to a common length (optionally rounded up to pad_to_multiple_of).
        """
        input_ids = tokenized_examples['input_ids']
        attention = tokenized_examples['attention_mask']
        # Find the longest sequence
        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            # Round up to nearest multiple for tensor core efficiency
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        padded_ids = []
        padded_mask = []
        pad_id = self.tokenizer.pad_token_id
        for ids, mask in zip(input_ids, attention):
            ids = ids + [pad_id] * (max_len - len(ids))
            mask = mask + [0] * (max_len - len(mask))
            padded_ids.append(torch.tensor(ids))
            padded_mask.append(torch.tensor(mask))
        return {
            'input_ids': torch.stack(padded_ids),      # shape (batch_size, max_len)
            'attention_mask': torch.stack(padded_mask) # shape (batch_size, max_len)
        }

    def __call__(self, features):
        """
        `features` is a list of examples, each example a dict with keys
        'anchor', 'positives', and 'negatives'.  We return a list of three
        dicts, one per role.
        """
        anchors   = [feat['anchor']    for feat in features]
        positives = [feat['positives'] for feat in features]
        negatives = [feat['negatives'] for feat in features]

        # Tokenize without padding; we will pad per-role
        tok_anchors   = self.tokenizer(anchors,   padding=False, truncation=True)
        tok_positives = self.tokenizer(positives, padding=False, truncation=True)
        tok_negatives = self.tokenizer(negatives, padding=False, truncation=True)

        # Pad each role separately
        batch_anchor   = self._pad_batch(tok_anchors)
        batch_positive = self._pad_batch(tok_positives)
        batch_negative = self._pad_batch(tok_negatives)

        # Return in the order (anchor, positive, negative)
        return [batch_anchor, batch_positive, batch_negative]
