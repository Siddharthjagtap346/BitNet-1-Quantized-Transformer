
import torch

PAD_TOKEN_ID = 0

def collate_fn(batch, pad_token_id=PAD_TOKEN_ID):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    function_labels = [item["function_labels"] for item in batch]
    domain_labels = [item["domain_labels"] for item in batch]
    loc_labels = [item["loc_labels"] for item in batch]
    go_labels = [item["go_labels"] for item in batch]

    max_len = max(len(x) for x in input_ids)
    batch_size = len(batch)

    padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        seq_len = len(inp)
        padded_inputs[i, :seq_len] = inp
        padded_labels[i, :seq_len] = lab
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask,
        "function_labels": torch.stack(function_labels),
        "domain_labels": torch.stack(domain_labels),
        "loc_labels": torch.stack(loc_labels),
        "go_labels": torch.stack(go_labels)
    }
