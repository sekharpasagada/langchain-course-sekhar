import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

# LangSmith tracing configuration for LangChain observability.
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or os.getenv(
    "LANGSMITH_API_KEY", ""
)
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "langchain-course-sekhar"
)


def main():
    print("Hello from langchain-course-sekhar!")

    # Load the Google API key from environment variables.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to a .env file or your environment."
        )

    # Source text used for the prompt.
    information = """
    Albert Einstein[a] (14 March 1879 – 18 April 1955) was a German-born theoretical physicist best known for developing the theory of relativity. Einstein also made important contributions to quantum theory.[1][5] His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called "the world's most famous equation".[6] He received the 1921 Nobel Prize in Physics for "his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect".
    """

    # Prompt asking the model to summarize the person and list interesting facts.
    summary_template = """
    Given this information about a person: {information}

    Create:
    1. A short summary
    2. Two interesting facts about the person
    """

    # Bind the template to the input variable so it can be used in a LangChain pipeline.
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Create the Google Gemini chat model using the API key and a low temperature for concise answers.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.3,
    )

    # Local Ollama model configuration for a local LLM run.
    """llm = ChatOllama(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        temperature=0.3,
    )"""

    # Connect the prompt template to the model and run the chain.
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
