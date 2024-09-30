"""
AI-powered Book Generator using Google's Gemini API

This script generates a complete book, including plot, chapters, title, and cover image,
based on user-provided parameters. It uses the Gemini API for text generation and
the Stability API for cover image creation.

Author: AI Assistant
Date: 2024-09-30
Dependencies: google-generativeai, ebooklib, requests
"""

import google.generativeai as genai
import os
import time
import re
from ebooklib import epub
import base64
import requests
import argparse

# Configure Gemini API
genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro-exp-0827")

# Stability API key for cover image generation
stability_api_key = "sk-AEVSg3punakzTFShSm5dPFQ3eiXLOeeBDgvfUy7tSvU2Ui5R"  # get it at https://beta.dreamstudio.ai/

def remove_first_line(test_string):
    """
    Remove the first line of the string if it starts with "Here" and ends with ":".

    Args:
        test_string (str): The input string to process.

    Returns:
        str: The processed string with the first line removed if it matches the condition.
    """
    if test_string.startswith("Here") and test_string.split("\n")[0].strip().endswith(":"):
        return re.sub(r'^.*\n', '', test_string, count=1)
    return test_string

def generate_text(prompt, max_tokens=2000, temperature=0.7):
    """
    Generate text using the Gemini API.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness in generation. Higher values make output more random.

    Returns:
        str: The generated text, with the first line removed if necessary.
    """
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature
    ))
    return remove_first_line(response.text)

def generate_cover_prompt(plot):
    """
    Generate a prompt for creating a book cover based on the plot.

    Args:
        plot (str): The plot of the book.

    Returns:
        str: A two-sentence description for the book cover.
    """
    response = generate_text(f"Plot: {plot}\n\n--\n\nDescribe the cover we should create, based on the plot. This should be two sentences long, maximum.")
    return response

def generate_title(plot):
    """
    Generate a title for the book based on its plot.

    Args:
        plot (str): The plot of the book.

    Returns:
        str: The generated title.
    """
    response = generate_text(f"Here is the plot for the book: {plot}\n\n--\n\nRespond with a great title for this book. Only respond with the title, nothing else is allowed.")
    return remove_first_line(response)

def create_cover_image(plot):
    """
    Create a cover image based on the plot using the Stability API.

    Args:
        plot (str): The plot outline of the book.

    Raises:
        Exception: If the Stability API key is missing or the API response is not successful.
    """
    # Generate the cover description from the plot
    plot = str(generate_cover_prompt(plot))

    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = stability_api_key

    if api_key is None:
        raise Exception("Missing Stability API key.")

    # Make a request to the Stability API to generate the cover image
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": plot
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 768,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    # Decode and save the generated image
    for i, image in enumerate(data["artifacts"]):
        with open("cover.png", "wb") as f:  # Update the path if running locally
            f.write(base64.b64decode(image["base64"]))

def generate_chapter_title(chapter_content):
    """
    Generate a title for a chapter based on its content.

    Args:
        chapter_content (str): The content of the chapter.

    Returns:
        str: The generated chapter title.
    """
    response = generate_text(f"Chapter Content:\n\n{chapter_content}\n\n--\n\nGenerate a concise and engaging title for this chapter based on its content. Respond with the title only, nothing else.")
    return remove_first_line(response)

def create_epub(title, author, chapters, cover_image_path='cover.png'):
    """
    Create an EPUB file from the generated book content.

    Args:
        title (str): The title of the book.
        author (str): The author of the book.
        chapters (list): A list of chapter contents.
        cover_image_path (str, optional): Path to the cover image. Defaults to 'cover.png'.
    """
    book = epub.EpubBook()
    
    # Set metadata
    book.set_identifier('id123456')
    book.set_title(title)
    book.set_language('en')
    book.add_author(author)
    
    # Add cover image
    with open(cover_image_path, 'rb') as cover_file:
        cover_image = cover_file.read()
    book.set_cover('cover.png', cover_image)
    
    # Create chapters and add them to the book
    epub_chapters = []
    for i, chapter_content in enumerate(chapters):
        chapter_title = generate_chapter_title(chapter_content)
        chapter_file_name = f'chapter_{i+1}.xhtml'
        epub_chapter = epub.EpubHtml(title=chapter_title, file_name=chapter_file_name, lang='en')
        
        # Add paragraph breaks
        formatted_content = ''.join(
            f'<p>{paragraph.strip()}</p>' 
            for paragraph in chapter_content.split('\n') 
            if paragraph.strip()
        )
        epub_chapter.content = f'<h1>{chapter_title}</h1>{formatted_content}'
        book.add_item(epub_chapter)
        epub_chapters.append(epub_chapter)
    
    # Define Table of Contents
    book.toc = tuple(epub_chapters)

    # Add default NCX and Nav files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define CSS style
    style = '''
    @namespace epub "http://www.idpf.org/2007/ops";
    body {
        font-family: Cambria, Liberation Serif, serif;
    }
    h1 {
        text-align: left;
        text-transform: uppercase;
        font-weight: 200;
    }
    '''

    # Add CSS file
    nav_css = epub.EpubItem(
        uid="style_nav", 
        file_name="style/nav.css", 
        media_type="text/css", 
        content=style
    )
    book.add_item(nav_css)

    # Create spine
    book.spine = ['nav'] + epub_chapters

    # Save the EPUB file
    epub.write_epub(f'{title}.epub', book)

def generate_book(writing_style, book_description, num_chapters):
    """
    Generate a complete book based on the given parameters.

    Args:
        writing_style (str): The desired writing style for the book.
        book_description (str): A high-level description of the book.
        num_chapters (int): The number of chapters to generate.

    Returns:
        tuple: A tuple containing the plot outline, full book text, and a list of chapters.
    """
    print("Generating plot outline...")
    plot_prompt = (
        f"Create a comprehensive and detailed plot outline for a {num_chapters}-chapter book "
        f"in the {writing_style} genre. The outline should be based on the following description:\n\n"
        f"{book_description}\n\n"
        f"Ensure that each chapter outline includes key events, character developments, and maintains a coherent narrative flow. "
        f"Each chapter should be designed to span approximately 10 pages."
    )
    
    plot_outline = generate_text(plot_prompt)
    print("Plot outline generated.")

    chapters = []
    for i in range(num_chapters):
        print(f"Generating chapter {i+1}...")
        chapter_prompt = (
            f"Based on the following plot outline and previous chapters, write chapter {i+1} of the book in the {writing_style} style. "
            f"The chapter should adhere to the plot outline and build upon the developments in previous chapters.\n\n"
            f"Plot Outline:\n{plot_outline}\n\n"
            f"Previous Chapters:\n{' '.join(chapters)}\n\n"
            f"Chapter {i+1} Content:"
        )
        chapter = generate_text(chapter_prompt, max_tokens=4000)
        chapters.append(remove_first_line(chapter))
        print(f"Chapter {i+1} generated.")
        time.sleep(1)  # Add a short delay to avoid hitting rate limits

    print("Compiling the book...")
    book = "\n\n".join(chapters)
    print("Book generated!")

    return plot_outline, book, chapters

def main(writing_style, book_description, num_chapters):
    """
    Main function to orchestrate the book generation process.

    Args:
        writing_style (str): The desired writing style for the book.
        book_description (str): A high-level description of the book.
        num_chapters (int): The number of chapters to generate.
    """
    # Generate the book
    plot_outline, book, chapters = generate_book(writing_style, book_description, num_chapters)

    title = generate_title(plot_outline)

    # Save the book to a file
    with open(f"{title}.txt", "w") as file:
        file.write(book)

    create_cover_image(plot_outline)

    # Create the EPUB file
    create_epub(title, 'AI', chapters, '/content/cover.png')

    print(f"Book saved as '{title}.txt' and '{title}.epub'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a book using AI")
    parser.add_argument("--style", required=True, help="The desired writing style")
    parser.add_argument("--description", required=True, help="A high-level description of the book")
    parser.add_argument("--chapters", type=int, required=True, help="The number of chapters")

    args = parser.parse_args()

    main(args.style, args.description, args.chapters)