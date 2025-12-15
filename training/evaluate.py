
diffs = []

model.eval()

for i in range(100):
    # 1. Sample from dataset
    s = robot_ds[random.randint(0, len(robot_ds) - 1)]

    image = s["image"].unsqueeze(0)        # (1, 3, 224, 224)
    input_ids = s["input_ids"].unsqueeze(0)

    # 2. Build attention mask manually (dataset does NOT store it)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 3. Forward pass
    with torch.no_grad():
        out = model(
            images=image,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # 4. Predicted action tokens (last 8 tokens)
    pred = out.logits[:, -8:].argmax(-1).squeeze(0)

    # 5. Ground-truth tokens come from labels, NOT action_tokens
    gt = s["labels"]
    gt = gt[gt != -100]          # remove padding / ignored tokens
    gt = gt[-8:]                 # keep only action tokens

    # 6. Token-level L1 difference
    diffs.append((pred - gt).abs().float().mean().item())

print("Mean token L1 diff:", sum(diffs) / len(diffs))
