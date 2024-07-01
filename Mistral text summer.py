from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: If you're using a private model, you may need to authenticate
# notebook_login()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_text(text, max_length=150, min_length=40):
    """
    Summarizes the given text using Mistral 7B model.

    Args:
    text (str): The input text to be summarized.
    max_length (int): The maximum length of the summary.
    min_length (int): The minimum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Generate the summary
    summary = summarization_pipeline(text, max_length=max_length, min_length=min_length)

    # Extract the summary text from the result
    summary_text = summary[0]['summary_text']

    return summary_text

# Example usage:
input_text = """
    Insert your text here.
    This can be a long text that you want to summarize using Mistral 7B.
"""
print(summarize_text(input_text))