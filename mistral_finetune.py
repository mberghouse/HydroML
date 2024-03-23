import os
import tkinter as tk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Create the main window
window = tk.Tk()
window.title("Mistral Finetuning App")

# Function to select the dataset file
def select_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    dataset_entry.delete(0, tk.END)
    dataset_entry.insert(tk.END, file_path)

# Function to start the finetuning process
def start_finetuning():
    dataset_path = dataset_entry.get()
    model_name = "stanford-crfm/mistral-base"
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load and tokenize the dataset
    with open(dataset_path, "r") as file:
        dataset = file.read()
    tokenized_dataset = tokenizer(dataset, truncation=True, padding=True, return_tensors="pt")
    
    # Format the dataset for finetuning
    train_dataset = tokenized_dataset["input_ids"]
    
    # Set up the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start the finetuning process
    trainer.train()
    
    # Save the finetuned model
    model.save_pretrained("finetuned_mistral")
    
    # Run inference tests
    finetuned_model = AutoModelForCausalLM.from_pretrained("finetuned_mistral")
    
    test_questions = [
        "What is the capital of France?",
        "Who wrote the play Romeo and Juliet?",
        "What is the largest planet in our solar system?",
    ]
    
    for question in test_questions:
        input_ids = tokenizer.encode(question, return_tensors="pt")
        output = finetuned_model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Question: {question}\nResponse: {response}\n")

# Create the dataset selection UI
dataset_label = tk.Label(window, text="Select Dataset:")
dataset_label.pack()
dataset_entry = tk.Entry(window, width=50)
dataset_entry.pack()
dataset_button = tk.Button(window, text="Browse", command=select_dataset)
dataset_button.pack()

# Create the start button
start_button = tk.Button(window, text="Start Finetuning", command=start_finetuning)
start_button.pack()

# Run the main event loop
window.mainloop()