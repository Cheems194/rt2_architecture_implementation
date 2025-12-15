def collate_fn(batch, pad_token_id, eos_token_id):
    images = torch.stack([x["image"] for x in batch])

    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=eos_token_id
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    attention_mask = (input_ids != eos_token_id).long()

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
