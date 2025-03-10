import os
import openai
from dotenv import load_dotenv
import logging
import pandas as pd
from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DELAY_TIME = 30  # Delay to manage API rate limits
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o"]  # OpenAI models available
DEFAULT_MODEL = "gpt-4o-mini"  # Default model

# Global OpenAI client
client = None


def load_api_key() -> str:
    """Load the OpenAI API key from environment variables or .env file in the repository."""

    # Try loading from standard environment variables (e.g., .zshrc, .bashrc, or system envs)
    api_key = os.getenv('PERSONAL_OPENAI_KEY')

    if api_key:
        logger.info("API key loaded from environment variables.")
        return api_key

    # Fallback: Manually load from .env file in the current repo
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        logger.info(f"Attempting to load API key from {env_path}")
        load_dotenv(env_path)
        api_key = os.getenv('PERSONAL_OPENAI_KEY')

        if api_key:
            logger.info("API key loaded from .env file in repository.")
            return api_key

    # If still no key, raise an error
    error_msg = "Error: No API key found. Please ensure PERSONAL_OPENAI_KEY is set in the environment or .env file."
    logger.error(error_msg)
    raise ValueError(error_msg)


def initialize_openai_client() -> None:
    """Initialize the global OpenAI client."""
    global client
    api_key = load_api_key()

    try:
        client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized and validated successfully.")
    except Exception as e:
        error_msg = f"Invalid API key or connection issue: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)


# Initialize the OpenAI client at startup
initialize_openai_client()


def set_model(model, state):
    state[0] = model  # Update the state
    return f"Model set to {model}"


# Function to stream logs to output box
def log_output(message: str):
    logger.info(message)
    return message


def set_model(model, state):
    state[0] = model  # Update the state
    return f"Model set to {model}"


# Function to stream logs to output box
def log_output(message: str):
    logger.info(message)
    return message


def estimate_tokens(word_count):
    # Average token per word ratio
    avg_tokens_per_word = 1.33
    return int(word_count * avg_tokens_per_word)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def synthesize_discussions(model: str, discussions: List[str], context: str, word_count: int) -> str:
    prompt = f"""
    The goal of this synthesis is to condense multiple discussion posts into a clear and engaging paragraph suitable for a providing to students as a recap. 
    The final output should capture the key points, shared insights, and overarching themes from the discussions while maintaining a friendly and accessible tone for a broad audience.
    
    Frame the synthesis within the following context:
    Context: {context}
    
    <DISCUSSIONS>
    Discussions: {discussions}
    </DISCUSSIONS>
    
    Task:
    - Generate a cohesive and concise summary of the key takeaways from the discussion posts listed between <DISCUSSIONS></DISCUSSIONS>. 
    - Ensure it flows naturally, highlights any consensus or differing opinions, and aligns with the provided context. 
    - Be sure to include good examples from the students discussions.
    - Ensure the presentation is quick to digest (bullet points) and easy to understand, especially for students where english is not their first language.
    
    As a concluding point, you must give your best answer to the context and discussion in one sentence, using the format- "LLM Answer:"
    
    Keep the response under {word_count} words.
    """

    time.sleep(DELAY_TIME)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that creates comprehensive, well-structured summaries"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=estimate_tokens(word_count)
    )
    return response.choices[0].message.content


def process_discussion_in_gradio(model: str, excel_file: str, context: str, word_count: int, format: str):

    filename = os.path.basename(excel_file)
    log_output(f"Processing: {filename}")

    ##code to get list of discussions

    if format == "Padlet":
        df = pd.read_excel(f'./Discussions/{filename}')
        summaries = [discussion for discussion in df['Body'] if discussion != 'No data']
    else:
        df = pd.read_csv(f'./Discussions/{filename}')
        summaries = [discussion for discussion in df.index[1:]]


    discussion_summary = synthesize_discussions(model, summaries, context, word_count)
    output_filename = f'./Summaries/{filename}_summary.txt'

    with open(output_filename, 'w') as f:
        f.write(discussion_summary)

    log_output(f"Discussion summary completed and saved as {output_filename}")

    return discussion_summary


def main():
    with gr.Blocks(
    theme='ParityError/Interstellar', #Add custom theme (purple)
    css="""
        #discussion-box {height: 200px;}
        #selection-box {height: 92px;}
        #output-box {height: 615px; overflow: auto;}
    """) as demo:
        gr.Markdown("# Discussion Summariser")

        with gr.Row():
            # Left Column
            with gr.Column(scale=1):
                with gr.Row():
                    excel_input = gr.File(
                        label="Upload Discussion",
                        file_types=[".xlsx", ".csv"],
                        file_count="single",
                        elem_id="discussion-box"
                    )

            # Left Column
            with gr.Column(scale=1):
                word_count_slider = gr.Slider(
                    label="Word Count Limit",
                    minimum=100,
                    maximum=500,
                    step=50,
                    value=250,
                    interactive=True,
                    elem_id='selection-box'
                )
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="Select OpenAI Model",
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_MODEL,  # Default selection
                        interactive=True,
                        elem_id='selection-box'
                    )
                    run_button = gr.Button("Generate", elem_id='selection-box')
        with gr.Row():
            # Left Column
            with gr.Column(scale=1):
                with gr.Row():
                    format_checkbox = gr.Radio(["Padlet", "H5P"], label="Format", info="Which format is the discussion in?", value='Padlet')

                context_textbox = gr.Textbox(
                    label="Edit Context",
                    lines=20,
                    placeholder="Context description will appear here for editing.",
                    interactive=True,
                    elem_id="context-box"
                )

            # Right Column
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Output",
                    lines=30,
                    elem_id="output-box"
                )

        def check_model_name(model_name):
            """Load the content of the selected task into the textbox."""
            log_output(f"Changed model to {model_name}")

        def process_files_and_task(model, excel_files, context, word_count, format):
            return process_discussion_in_gradio(model, excel_files, context, word_count, format)



        # Populate the textbox when a task is selected
        model_dropdown.change(fn=check_model_name, inputs=model_dropdown, outputs=None)

        # Use the content of the textbox for processing
        run_button.click(fn=process_files_and_task,
                         inputs=[model_dropdown, excel_input, context_textbox, word_count_slider, format_checkbox], outputs=output)

    demo.launch()


if __name__ == "__main__":
    main()
