
class MiniRT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vision_model_name: str = "google/vit-base-patch16-224",
        d_model: int = 768,
        n_layer: int = 8,
        n_head: int = 8,
        max_seq_len: int = 128,
    ):
        super().__init__()

        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_encoder.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        vision_dim = self.vision_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_dim, d_model)

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_seq_len + 256,
            bos_token_id=None,
            eos_token_id=None,
        )

        self.decoder = GPT2LMHeadModel(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        images,
        input_ids,
        attention_mask=None,
        labels=None
    ):
        """
        images: (B, 3, 224, 224)
        input_ids: (B, T)
        attention_mask: (B, T)
        labels: (B, T)
        """

        B = images.size(0)

        with autocast(enabled=False):
            images = images.float()
            vision_outputs = self.vision_encoder(pixel_values=images)
            vision_embeds = self.vision_proj(
                vision_outputs.last_hidden_state
            )

        token_embeds = self.decoder.transformer.wte(input_ids)

        inputs_embeds = torch.cat(
            [vision_embeds, token_embeds], dim=1
        )

        if attention_mask is not None:
            vision_mask = torch.ones(
                (B, vision_embeds.size(1)),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat(
                [vision_mask, attention_mask], dim=1
            )

        if labels is not None:
            vision_label_pad = torch.full(
                (B, vision_embeds.size(1)),
                -100,
                device=labels.device,
                dtype=labels.dtype
            )
            labels = torch.cat(
                [vision_label_pad, labels], dim=1
            )

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )

        if labels is not None:
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            min_len = min(
                shift_logits.size(1),
                shift_labels.size(1)
            )

            shift_logits = shift_logits[:, :min_len, :].contiguous()
            shift_labels = shift_labels[:, :min_len].contiguous()

            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            outputs.loss = loss

        return outputs
