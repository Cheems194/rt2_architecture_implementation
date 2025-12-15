model.train()
global_step = 0

for epoch in range(EPOCHS):

    robot_ratio = get_robot_ratio(epoch)
    print(f"Epoch {epoch+1}: robot_ratio={robot_ratio}")

    sampler = MixedBatchSampler(
        robot_len=len(robot_ds),
        vlm_len=len(vlm_ds),
        batch_size=BATCH_SIZE,
        robot_ratio=robot_ratio
    )

    loader = DataLoader(
    combined_ds,
    batch_sampler=sampler,
    collate_fn=lambda b: collate_fn(
        b,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    ),
    num_workers=0
)


    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for batch in pbar:
        if global_step >= MAX_STEPS:
            break

        images = batch["images"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "step": global_step
        })

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step
    }, f"checkpoint_epoch_{epoch+1}.pt")

    if global_step >= MAX_STEPS:
        break
