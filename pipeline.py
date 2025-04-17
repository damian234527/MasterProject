from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load
import torch

# Step 1: Load Dataset
# Replace with actual dataset loading logic
dataset = load_dataset("webis_clickbait17")  # Example dataset

def preprocess_function(examples):
    return tokenizer(examples["headline"], examples["article"], truncation=True, padding="max_length", max_length=512)

# Step 2: Load Pretrained Model and Tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = dataset.map(preprocess_function, batched=True)

# Step 3: Training Setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Step 4: Define Evaluation Function
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions.numpy(), references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 5: Train Model
trainer.train()

# Step 6: Save Model
trainer.save_model("./saved_model")

# Step 7: Convert to ONNX for Deployment
import torch.onnx

dummy_input = torch.randint(0, 100, (1, 512))
torch.onnx.export(model, dummy_input, "model.onnx")
