import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the LangChain tracer for LangSmith
tracer = LangChainTracer()


def syntactical_quiz_agent(tech_summary, no_of_questions, llm):
    prompt_template = f"""
    Role: You are a creative AI assistant specializing in generating programming-related educational materials. Your task is to create multiple-choice quizzes that are syntax-focused, highly accurate, and based on a provided {{tech_summary}}. Your goal is to ensure the quiz is clear, unambiguous, and effectively tests the learner’s understanding of the provided technical concepts.

    Instructions:
    1. Analyze Input: Carefully read and understand the provided technical summary.
    2. Create Questions: Based on the summary, create a multiple-choice quiz with {{no_of_questions}} MCQs. Each question should be syntax-based and may have one or more correct answers. The quiz should focus on:
        - Code snippets or syntax-related concepts.
        - Common pitfalls or best practices related to the described topic.
    3. Answer Validation: For each question:
        - Include 4 options (label them as A, B, C, D).
        - Clearly indicate which option(s) is/are correct. If multiple answers are correct, specify all correct answers.
    4. Check for Ambiguity: Ensure all questions and options are clear, unambiguous, and directly tied to the summary. Avoid vague or overly broad options.
    5. Structure the Output: Present the quiz in the following format:
        - **Question 1:** [Question Text]
            - A. [Option A]
            - B. [Option B]
            - C. [Option C]
            - D. [Option D]
        **Answer:** [Correct Option(s)]
        Repeat this format for all questions.

    6. General Guidelines:
        - Make the questions challenging but not excessively difficult.
        - Avoid creating options that are too similar or misleading.
        - Ensure all correct answers are aligned with the content of the technical summary.

    Output Example:
    Here's an example of the desired output format:

    **Question 1:** Which of the following are valid ways to create a list in Python?
    - A. `my_list = list([1, 2, 3])`
    - B. `my_list = {1, 2, 3}`
    - C. `my_list = [1, 2, 3]`
    - D. `my_list = (1, 2, 3)`
    **Answer:** A, C
    """

    prompt = PromptTemplate.from_template(prompt_template)

    formatted_prompt = prompt.format(
        tech_summary=tech_summary,
        no_of_questions=no_of_questions
    )

    model = ChatOpenAI(model=llm)

    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response
