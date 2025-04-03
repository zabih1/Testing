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



def tech_summary_agent(skill_track, skill, objective, difficulty, llm):
    """Provides a difficulty-level technical summary for a given skill."""

    prompt_template = f"""
    You are an AI tutor specializing in the {skill_track} track. Your task is to provide a **technical summary** for the topic: {skill}, with the specific objective: 
    {objective}. 

    The summary must:  
    1. Be written at a {difficulty} level to match the learner's expertise.  
    2. Focus strictly on the given skill, objective, and skill track without deviating from them.  
    3. Include:  
    - Key technical concepts.  
    - Critical details relevant to the topic.  
    - Practical applications that demonstrate how the skill is used.  

    Ensure the explanation is precise, objective-driven, and technical, while remaining appropriate for the specified {difficulty} level.
    """

    prompt = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(
        skill_track=skill_track,
        skill=skill,
        objective=objective,
        difficulty=difficulty
    )
    
    model = ChatOpenAI(model=llm)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response



def fundamental_summary_agent(summary, llm):
    """Rewrites the technical summary focusing on fundamental concepts."""

    prompt_template = f"""
    Below is a detailed summary of a specific skill:

    {summary}

    Re-write this summary to focus exclusively on the **fundamental concepts** of the skill. Remove any references to specific programming languages, tools, or syntax. The revised summary should:  
    1. Be general and applicable across any programming environment or context.  
    2. Highlight the core principles, techniques, and concepts essential to understanding the skill.  
    3. Avoid unnecessary technical details or implementation specifics.  

    Ensure the explanation remains clear, conceptual, and accessible to anyone learning the foundational aspects of this skill.
    """

    prompt = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(summary=summary)
    model = ChatOpenAI(model=llm)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response



def fundamental_quiz_agent(fundamental_summary, number_of_quiz, llm):
    """Generates quiz questions based on a fundamental summary."""

    prompt_template = f"""
    Role: You are an AI tutor specializing in creating conceptual quizzes. Based on the topic: {fundamental_summary}, generate {number_of_quiz} multiple-choice questions (MCQs).

    Instructions:
    1. Analyze Input: Carefully read and understand the provided fundamental summary.
    2. Create Questions: Based on the summary, create a multiple-choice quiz with {number_of_quiz} MCQs. The questions should be language-agnostic, focusing on core principles and ideas rather than programming syntax or language-specific features. Each question may have one or more correct answers.
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

    """

    prompt = PromptTemplate.from_template(prompt_template)

    formatted_prompt = prompt.format(
        fundamental_summary=fundamental_summary,
        number_of_quiz=number_of_quiz
    )
    model = ChatOpenAI(model=llm)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response


