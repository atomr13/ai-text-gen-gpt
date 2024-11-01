# GPT-2 Text Generator with Gradio

This project is a simple text generation application that uses GPT-2, a pre-trained language model by OpenAI, and Gradio, a Python library for creating user-friendly web interfaces. Users can input any prompt and the model will generate text based on that input.

## Overview

The project leverages the capabilities of the GPT-2 model from the `transformers` library to generate coherent and meaningful text given an input prompt. Gradio is used to create a web-based interface, allowing users to easily interact with the model without any complex setup.

## Requirements

To run this project, you will need the dependencies listed in the `requirements.txt` file.

You can install the dependencies using pip:

```sh
pip install -r requirements.txt
```

## How It Works

The script loads the GPT-2 model and tokenizer from Hugging Face's `transformers` library. The Gradio interface takes user input as a text prompt, generates a response using GPT-2, and displays the generated text.

### Text Generation Parameters

- **Prompt**: The input text provided by the user that acts as a starting point for the generated text.
- **Temperature**: Controls the randomness of the output. Higher values make the text more diverse, while lower values make it more focused and deterministic.
- **Max Length**: Defines the maximum number of tokens the generated text can contain.

## Running the Project

1. Clone this repository or copy the code to a local Python file.
2. Ensure you have installed all the required dependencies.
3. Run the script using Python:

```sh
python app.py
```

4. After running the script, a local Gradio interface will open in your browser, where you can enter prompts and get generated responses from GPT-2.

## Example Usage

Once the script is running, open the link provided by Gradio in your browser. Enter a prompt, such as:

```
Once upon a time, in a faraway land
```

And click the "Generate Text" button to see what GPT-2 generates next!

## Code Overview

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Text generation function
def generate_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
    return tokenizer.batch_decode(gen_tokens)[0]

# Gradio Interface
interface = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="GPT-2 Text Generator")

# Launch Interface
interface.launch()
```

## Technologies Used

- **Python**: Programming language used to implement the text generation.
- **Transformers Library**: Hugging Face's library for accessing pre-trained models like GPT-2.
- **Gradio**: Library for creating web interfaces for machine learning models.
- **PyTorch**: Framework used to run the GPT-2 model.

## Future Improvements

- **User-Adjustable Parameters**: Allow users to modify generation parameters such as temperature and max length via the web interface.
- **Model Customization**: Enable fine-tuning the model on specific domains or topics.
- **Deploying Online**: Deploy the interface to a public server (using platforms like Hugging Face Spaces or Streamlit Sharing).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- **OpenAI** for developing GPT-2.
- **Hugging Face** for making pre-trained models easily accessible.
- **Gradio** for simplifying the process of creating web interfaces for ML models.

