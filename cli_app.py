import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment variable
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Function to generate AI response
def generate_summary(user_input):
    response = openai.ChatCompletion.create(
        model = "openai/gpt-3.5-turbo",
        messages = [
            {"role":"system", "content":"You are an AI assistant that summarizes and formats text."},
            {"role":"user", "content": f"Summarize this text:\n{user_input}"}
        ],
        max_tokens = 200,
        temperature = 0.5
    )
    return response.choices[0].message["content"]

# Main script
if __name__ == "__main__": # This line means program is directly running(not through importing)
    print("AI Automation Tool - Phase 1")
    print("Type/paste any text and get a summary!\n")

    while True:
        user_text = input("Enter your text (or type 'exit' to quit): ")

        if user_text.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        ai_output = generate_summary(user_text)
        print("\n AI Generated Summary:\n", ai_output)
        print("\n" + "-"*50 + "\n") 

    