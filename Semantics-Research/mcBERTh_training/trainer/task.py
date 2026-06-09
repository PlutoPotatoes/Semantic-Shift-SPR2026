from transformers import (
    BertForMaskedLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    TrainerCallback
)
from data_streamer import build_decade_balanced_stream, DECADES
import os
import json
import math
from google.cloud import storage
from google.oauth2 import service_account
import math
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import sys
import gcsfs

# ── Hyperparameters ───────────────────────────────────────────────
model_name = "emanjavacas/MacBERTh"
experiment_name = "McBERTh-Pretrain-v1"
epochs = 2
learning_rate = 5e-5
batch_size = 32
# gradient_accumulation_steps = (batchsize * 8) / (batchsize * #_GPU)
# 1 GPU:  256 / (32 * 1) = 8
# 2 GPUs: 256 / (32 * 2) = 4
gradient_accumulation_steps = 8
max_steps = math.ceil(2102849 / (batch_size * gradient_accumulation_steps)) * epochs
max_steps = 50 # for testing only, comment out for full training
logging_steps = 100
warmup_ratio = 0.05
weight_decay = 0.01
save_steps = 500
mlm_probability = 0.15
gcs_credentials = "nlp-research-sp26-8499634f1c62.json"

# ── CUDA check ─────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: CUDA not available. Aborting training job.")
    sys.exit(1)

# ── Bucket check ─────────────────────────────────────────────────────────
fs = gcsfs.GCSFileSystem()
print("Checking access to GCS bucket...")
if fs.exists("project3102-model-bucket/Training-Tests/"):
    print("Access to GCS bucket confirmed.")
    bucket_name = "project3102-model-bucket/Training-Tests/"
else:
    print("ERROR: Cannot access GCS bucket. Aborting training job.")
    sys.exit(1)

# ── Tokenizer ─────────────────────────────────────────────────────
# Here are are mostly just using the tokenizer our model already uses. 
# However, were going add special tokens for each decade our data belongs to 
# this is how our model will differentiate words used in different time periods
# ex. <decade_1990>
print("Start script")

print('Building tokenizer')
def get_date_tokens(decades):
    return [f"<decade_{str(d).removesuffix('s')}>" for d in decades]


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {'additional_special_tokens': get_date_tokens(DECADES)})


def tokenize_data(examples):
    result = tokenizer(examples["text"], max_length=512, truncation=True)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(
            i) for i in range(len(result["input_ids"]))]
    return result



# ── Datasets ──────────────────────────────────────────────────────
# COHA requires a license so we can't store it locally — shards are downloaded
# from GCS to the VM's local disk at startup, then streamed during training.
# Make sure the service account JSON is present in the container.
print("building dataset")

train_dataset = build_decade_balanced_stream(
    service_account_path=gcs_credentials)
# No need to shuffle validation set
val_dataset = build_decade_balanced_stream(
    service_account_path=gcs_credentials, split='valid', shuffle=False)
train_dataset = train_dataset.map(
    tokenize_data, batch_size=batch_size, batched=True)
val_dataset = val_dataset.map(
    tokenize_data,   batch_size=batch_size, batched=True)
print("dataset complete, formatting model")

# ── Model ─────────────────────────────────────────────────────────
# Defining our base model means downloading from huggingface and adding in our new decade tokens.
# we are also going to add in a callback function that helps it run smoother on google cloud
model = BertForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
)


class ContiguousParamsCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if not param.data.is_contiguous():
                print(f"[WARNING] Non-contiguous tensor found at save: {name}")
                param.data = param.data.contiguous()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Check every N steps to narrow down when it happens
        if state.global_step % 100 == 0:
            for name, param in model.named_parameters():
                if not param.data.is_contiguous():
                    print(f"[Step {state.global_step}] Non-contiguous: {name}")


# ── Training ──────────────────────────────────────────────────────
# Set all of our training parameters, define our Trainer object, start training.

# Vertex AI sets AIP_MODEL_DIR automatically — use it as output dir
print('Training start:')

output_dir = os.environ.get("AIP_MODEL_DIR", f"Models/{experiment_name}")
if not output_dir.startswith("gs://"):
    print(f"ERROR: output_dir is not a GCS path: {output_dir}")
    sys.exit(1)
print(f"Output directory: {output_dir}")


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_steps=max_steps,
    eval_strategy='steps',
    eval_steps=save_steps,
    save_strategy='steps',
    save_steps=save_steps,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    logging_steps=logging_steps,
    warmup_ratio=warmup_ratio,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim='adamw_torch',
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001), ContiguousParamsCallback()]
)

trainer.train()
print("training complete")

# ── Save ──────────────────────────────────────────────────────────
print("saving model")
save_path = f"{output_dir}/best"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")

# Save parameters JSON to GCS
train_loss = [e['loss'] for e in trainer.state.log_history if 'loss' in e]
eval_loss = [e['eval_loss']
             for e in trainer.state.log_history if 'eval_loss' in e]

params = {
    "model_name": model_name,
    "max_steps": max_steps,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "warmup_ratio": warmup_ratio,
    "weight_decay": weight_decay,
    "training_loss": train_loss,
    "eval_loss": eval_loss,
}

# ── Save parameters ──────────────────────────────────────────────────────────
credentials = service_account.Credentials.from_service_account_file(
    gcs_credentials)
client = storage.Client(credentials=credentials)
bucket = client.bucket("project3102-model-bucket")
blob = bucket.blob(f"Training-Tests/{experiment_name}/parameters.json")
blob.upload_from_string(json.dumps(params, indent=4),
                        content_type="application/json")
print("Parameters saved to GCS.")


