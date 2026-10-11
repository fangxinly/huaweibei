"""Exact optimizer function, statically extracted from original SHA acc37dce."""
def optimizer_for(author, model, steps):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    parameters = list(model.named_parameters())
    groups = [{"params": [p for n, p in parameters if not any(x in n for x in no_decay)],
               "weight_decay": .01},
              {"params": [p for n, p in parameters if any(x in n for x in no_decay)],
               "weight_decay": 0.0}]
    optimizer = author.AdamW(groups, lr=author.args.learning_rate)
    scheduler = author.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(author.args.warmup_proportion * steps),
        num_training_steps=steps)
    return optimizer, scheduler
