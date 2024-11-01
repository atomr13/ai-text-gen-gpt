from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load model and tokenizer
# GPT-2 is free to use so don't worry
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Text generation function
def generate_text(prompt):
    '''
    This function takes a text prompt as input, generates text using GPT-2, 
    and returns the generated text.
    
    Args:
        prompt (str): The input text to prompt the model.

    Returns:
        str: The generated text based on the given prompt.
    '''
    
    # Tokenize the input prompt and convert it to input IDs that the model understands
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids 

    # Generate text using the model
    # Parameters:
    # - do_sample=True: Enables sampling to introduce variability in output
    # - temperature=0.9: Controls randomness in text generation (higher is more random)
    # - max_length=100: Sets the maximum length of the generated text
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)

    # Decode the generated tokens back to a human-readable string
    return tokenizer.batch_decode(gen_tokens)[0]

# Gradio Interface
# Setting up a Gradio interface to make the text generator interactive
interface = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="GPT-2 Text Generator")

# Launch the Gradio Interface
# This will create a local server and provide a link to interact with the model through a web browser
interface.launch()






