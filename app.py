import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import pyttsx3
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from textblob import TextBlob

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize a question-answering pipeline without specifying a revision
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def read_text_from_file(filepath):
    """
    This function reads text content from a specified file path with error handling.
    """
    if not os.path.exists(filepath):
        messagebox.showerror("Error", "File not found!")
        return None
    try:
        with open(filepath, 'rb') as file:
            for encoding in ['utf-8', 'latin-1', 'windows-1252']:
                try:
                    text = file.read().decode(encoding)
                    return text
                except UnicodeDecodeError:
                    continue
            messagebox.showerror("Error", "Unable to decode the file using any of the supported encodings.")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        return None

def summarize_document(text, progress_var):
    """
    This function summarizes the entire document using the BART model.
    """
    max_chunk_length = 1024  # Define the desired maximum length size for input chunks
    text_chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    summarized_chunks = []
    for i, chunk in enumerate(text_chunks):
        input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=max_chunk_length, truncation=True)
        summary_ids = summarizer_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summarized_chunks.append(summary)
        progress_var.set((i + 1) / len(text_chunks) * 100)  # Update progress variable

    full_summary = " ".join(summarized_chunks)
    return full_summary

def browse_file(progress_var):
    """
    This function opens a file dialog for users to select the file.
    """
    filepath = filedialog.askopenfilename(title="Select File")
    if filepath:
        prompt_category_selection(filepath, progress_var)

def prompt_category_selection(filepath, progress_var):
    """
    This function prompts the user to select a document category.
    """
    category_window = tk.Toplevel(window)
    category_window.title("Select Document Category")

    label = tk.Label(category_window, text="Select Document Category:")
    label.pack(pady=5)

    category_var = tk.StringVar()
    category_dropdown = ttk.Combobox(category_window, textvariable=category_var)
    category_dropdown['values'] = ('Terms & Conditions', 'Rental & Lease Agreements', 'Privacy Policy', 'Employment Contract')
    category_dropdown.pack(pady=5)

    def on_category_selected():
        category = category_var.get()
        if category:
            analyze_document(filepath, progress_var, category)
            category_window.destroy()
        else:
            messagebox.showwarning("Warning", "Please select a document category.")

    select_button = tk.Button(category_window, text="Select", command=on_category_selected)
    select_button.pack(pady=5)

def analyze_document(filepath, progress_var, category):
    """
    This function analyzes the document by summarizing its content and displaying it in the GUI.
    """
    text = read_text_from_file(filepath)
    if text is None:
        return

    summary = summarize_document(text, progress_var)
    benefits, disadvantages = extract_benefits_and_disadvantages(summary, category)

    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, f"Summary:\n{summary}\n\nBenefits:\n{benefits}\n\nDisadvantages:\n{disadvantages}")

    sentiment_label.configure(text="Performing sentiment analysis...")
    sentiment_analysis_thread = threading.Thread(target=perform_sentiment_analysis, args=(summary,))
    sentiment_analysis_thread.start()

def extract_benefits_and_disadvantages(summary, category):
    """
    This function extracts benefits and disadvantages from the summarized text using contextual analysis.
    """
    questions = {
        "Terms & Conditions": {
            "benefits": "What are the benefits of agreeing to these terms and conditions?",
            "disadvantages": "What are the disadvantages of agreeing to these terms and conditions?"
        },
        "Rental & Lease Agreements": {
            "benefits": "What are the benefits of agreeing to this rental or lease agreement?",
            "disadvantages": "What are the disadvantages of agreeing to this rental or lease agreement?"
        },
        "Privacy Policy": {
            "benefits": "What are the benefits of agreeing to this privacy policy?",
            "disadvantages": "What are the disadvantages of agreeing to this privacy policy?"
        },
        "Employment Contract": {
            "benefits": "What are the benefits of agreeing to this employment contract?",
            "disadvantages": "What are the disadvantages of agreeing to this employment contract?"
        }
    }

    benefits = ""
    disadvantages = ""

    for key, question in questions[category].items():
        result = qa_pipeline(question=question, context=summary)
        if key == "benefits":
            benefits = result['answer']
        else:
            disadvantages = result['answer']

    return benefits, disadvantages

def perform_sentiment_analysis(summary):
    """
    This function performs sentiment analysis on the summary text.
    """
    blob = TextBlob(summary)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Beneficial"
        reason = "The clauses are mostly beneficial."
    elif polarity < 0:
        sentiment = "Risky"
        reason = "The clauses are not beneficial, beware."
    else:
        sentiment = "Neutral"
        reason = "The clauses are fair and treat well for both parties."

    sentiment_label.configure(text=f"Sentiment Analysis:\nSentiment: {sentiment}\nReason: {reason}")

def read_summary():
    """
    This function reads out the summarized text using text-to-speech engine.
    """
    summary_text = text_area.get("1.0", tk.END).strip()
    if not summary_text:
        messagebox.showwarning("Warning", "No summary to read!")
        return

    threading.Thread(target=speak_text, args=(summary_text,), daemon=True).start()

def speak_text(text):
    """
    This function speaks out the provided text using text-to-speech engine.
    """
    engine.say(text)
    engine.runAndWait()

def clear_text():
    """
    This function clears the text area.
    """
    text_area.delete("1.0", tk.END)

def main():
    global window
    window = tk.Tk()
    window.title("Document Summarizer")

    frame = tk.Frame(window)
    frame.pack()

    browse_button = tk.Button(frame, text="Browse", command=lambda: browse_file(progress_var))
    browse_button.pack(side=tk.LEFT, padx=5)

    read_button = tk.Button(frame, text="Read Summary", command=read_summary)
    read_button.pack(side=tk.LEFT, padx=5)

    clear_button = tk.Button(frame, text="Clear", command=clear_text)
    clear_button.pack(side=tk.LEFT, padx=5)

    global progress_var
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(window, variable=progress_var, maximum=100)
    progress_bar.pack(fill=tk.X)

    global text_area
    text_area = tk.Text(window, wrap=tk.WORD, width=60, height=20)
    text_area.pack(pady=10)
    scroll = tk.Scrollbar(window, command=text_area.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.config(yscrollcommand=scroll.set)

    global sentiment_label
    sentiment_label = tk.Label(window, text="File Synapsis:")
    sentiment_label.pack()

    window.mainloop()

if __name__ == "__main__":
    main()
