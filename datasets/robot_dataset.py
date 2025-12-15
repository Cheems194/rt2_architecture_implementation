class RobotDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, image_size=224):
        self.samples = []
        self.tokenizer = tokenizer

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.act_start_id = tokenizer.convert_tokens_to_ids("<act_start>")
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        image = self.transform(image)

        instr_ids = self.tokenizer.encode(
            sample["instruction"],
            add_special_tokens=False
        )
        input_ids = (
            [self.tokenizer.bos_token_id]
            + instr_ids
            + [self.act_start_id]
        )

        target_ids = sample["action_tokens"] + [self.eos_id]

        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
            "type": "robot"
        }
