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


def scenario_agent(field_name, llm ):
    
    prompt_template = f"""
    Role: You are a creative AI assistant specializing in designing realistic and engaging scenarios for educational purposes. Your task is to create scenarios that incorporate practical datasets to aid learning and analysis.

    Instruction:
    Create a detailed and realistic data analysis scenario based on the field: **{field_name}**.
    1. Provide a concise, engaging description of the scenario (2-3 lines) to set the context. The description should highlight the purpose and relevance of the dataset to the selected field.
    2. Include a sample dataset structure with relevant column names and 5 rows of example data. Ensure the dataset is realistic, meaningful, and directly tied to the context.

    Requirements:
    - The dataset should include key attributes commonly analyzed in the selected field.
    - Keep the scenario and dataset focused on analysis objectives, leaving room for exploration or question generation later.
    - Do **not** generate any questions—focus solely on the scenario and dataset.
    """
    prompt = PromptTemplate.from_template(prompt_template)

    formatted_prompt = prompt.format(field_name=field_name)
    
    model = ChatOpenAI(model=llm)

    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response



def scenario_quiz_agent(scenario_with_fake_data, tech_summary, number_of_quiz, llm):

    prompt_template = f"""
    Role: You are an advanced AI assistant specializing in creating high-quality, scenario-based multiple-choice questions (MCQs) for educational purposes. Your goal is to design engaging and thought-provoking MCQs that test understanding of a given scenario and technical summary.

    Instruction:
    Using the provided **scenario** and **technical summary**, create a set of {number_of_quiz} multiple-choice questions. The questions should be directly relevant to the context and objectives of the scenario and technical summary.

    Input Details:
    - Scenario:
    {scenario_with_fake_data}
    - Technical Summary:
    {tech_summary}

    Requirements:
    1. Question Design:
    - Each question should be scenario-driven and assess comprehension or problem-solving skills.
    - Each question should be technical.
    - Ensure the question is logically tied to the given scenario and technical summary.
    - Concept-based questions: Assess the learner's understanding of concepts described in the technical summary.
    - Code-based questions: Provide small code snippets or examples and ask questions about their output, functionality, or underlying logic.

    2. Options:
    - Provide four options for each question.
    - At least one option must be correct, but multiple correct answers are allowed. Clearly indicate the correct answers when providing output.

    3. Formatting:
    - Number each question.
    - Use clear, concise language for both questions and options.
    - Format as follows:
        ```
        Question Number: [Question text]
        a) [Option 1]
        b) [Option 2]
        c) [Option 3]
        d) [Option 4]
        
        Correct Answer(s): [a/b/c/d]
        ```

    4. Focus:
    - Avoid generic questions unrelated to the scenario.
    - Ensure the options are plausible and well-aligned with the context.

    Output Format:
    Return the questions in a structured and easy-to-read format as outlined above.
    """

            
    prompt = PromptTemplate.from_template(prompt_template)

    formatted_prompt = prompt.format(scenario_with_fake_data=scenario_with_fake_data, tech_summary=tech_summary, number_of_quiz=number_of_quiz)
    
    model = ChatOpenAI(model=llm)

    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response

