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
# logging_steps = 100
warmup_ratio = 0.05
weight_decay = 0.01
save_steps = 500
mlm_probability = 0.15
gcs_credentials = "nlp-research-sp26-8499634f1c62.json"

#temp parameters for quick testing
max_steps = 10
logging_steps = 2 
save_steps = 5
eval_steps = 5


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
    # word_ids are Python lists (not tensors) and not needed for MLM —
    # keeping them causes accelerate to hang when moving eval batches to GPU.
    return {k: v for k, v in result.items() if k != "word_ids"}



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
                print(f"[WARNING] Non-contiguous tensor found at save: {name}", flush=True)
            param.data = param.data.contiguous()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Check every N steps to narrow down when it happens
        if state.global_step % 100 == 0:
            for name, param in model.named_parameters():
                if not param.data.is_contiguous():
                    print(f"[Step {state.global_step}] Non-contiguous: {name}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = state.max_steps
        pct = (step / total * 100) if total else 0
        loss = logs.get("loss", "N/A")
        eval_loss = logs.get("eval_loss", None)
        lr = logs.get("learning_rate", "N/A")
        msg = f"[Step {step}/{total} | {pct:.1f}%] loss={loss} lr={lr}"
        if eval_loss is not None:
            msg += f" eval_loss={eval_loss}"
        print(msg, flush=True)


# ── Training ──────────────────────────────────────────────────────
# Set all of our training parameters, define our Trainer object, start training.

# Vertex AI sets AIP_MODEL_DIR automatically — use it as output dir
print('Training start:', flush=True)

output_dir = os.environ.get("AIP_MODEL_DIR", f"Models/{experiment_name}")
if not output_dir.startswith("gs://"):
    print(f"ERROR: output_dir is not a GCS path: {output_dir}", flush=True)
    sys.exit(1)
print(f"Output directory: {output_dir}", flush=True)

# Use a local logging dir — pointing logging_dir at gs:// can cause
# os.makedirs / file-writer initialization to block on Vertex AI.
local_logging_dir = "/tmp/tb_logs"
print(f"Local logging dir: {local_logging_dir}", flush=True)

print("Building TrainingArguments...", flush=True)
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
    logging_dir=local_logging_dir,
    logging_steps=logging_steps,
    warmup_ratio=warmup_ratio,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim='adamw_torch',
)
print("TrainingArguments built.", flush=True)

print("Building Trainer...", flush=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001), ContiguousParamsCallback()]
)
print("Trainer built.", flush=True)

print("starting training loop", flush=True)
trainer.train(resume_from_checkpoint=False)
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

credentials = service_account.Credentials.from_service_account_file(
    gcs_credentials)
client = storage.Client(credentials=credentials)
bucket = client.bucket("project3102-model-bucket")

# ── Save model──────────────────────────────────────────────────────────
gcs_model_prefix = f"Training-Tests/{experiment_name}/best"
for filename in os.listdir(save_path):
    local_file = os.path.join(save_path, filename)
    blob = bucket.blob(f"{gcs_model_prefix}/{filename}")
    blob.upload_from_filename(local_file)
    print(f"Uploaded {filename} to gs://project3102-model-bucket/{gcs_model_prefix}/{filename}")

print("Model upload to GCS")

# ── Save parameters ──────────────────────────────────────────────────────────
blob = bucket.blob(f"Training-Tests/{experiment_name}/parameters.json")
blob.upload_from_string(json.dumps(params, indent=4),
                        content_type="application/json")
print("Parameters saved to GCS.")

