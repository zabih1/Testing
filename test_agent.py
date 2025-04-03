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

model = ChatOpenAI(model="gpt-4")

def evaluating_conceptual_questions(agent_response):
    prompt = """
        Evaluate this AI agent response containing conceptual based MCQS against these strict criteria:
        1. **Accuracy**: Ensure there are no factual errors or hallucinations in the questions and answers.
        2. **Language-Agnostic Focus**: Check The questions should be language-agnostic, focusing on core principles and ideas rather than programming syntax.

        **Task:** Return ONLY "good" if all criteria are met, otherwise "not good". 

        **Forced Format:** 
        Final evaluation: [strictly two words]
        
        Agent's Response:
        {agent_response}
        """

    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.format(agent_response=agent_response)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response


def evaluating_scenario_questions(agent_response):
    prompt = """
            Evaluate this AI agent response containing scenario-based multiple-choice questions (MCQs) against the following strict criteria:

            1. **Accuracy**: Confirm that there are no factual errors or hallucinations in the questions, answer options, or correct answers.
            2. **Clarity**: Verify that the questions and answer options are clearly and concisely phrased.

            **Task:** Return ONLY "good" if all criteria are met, otherwise "not good".

            **Forced Format:** 
            Final evaluation: [strictly two words]

            Agent's Response:
            {agent_response}
        """

    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.format(agent_response=agent_response)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response


def evaluating_syntactical_questions(agent_response):
    prompt = """
            Evaluate this AI agent response containing programming-related, syntax-focused MCQs against the following strict criteria:

            1. **Syntax Accuracy**: Confirm that each question is strictly based on programming syntax and code snippets, and that any technical details or code examples are factually correct.
            2. **Clarity**: Ensure that the questions and answer options are clearly and concisely phrased, leaving no room for ambiguity.
            3. **Accuracy**: Ensure there are no factual errors or hallucinations in the questions and answers.

            **Task:** Return ONLY "good" if all criteria are met, otherwise "not good".

            **Forced Format:** 
            Final evaluation: [strictly two words]

            Agent's Response:
            {agent_response}
        """

    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.format(agent_response=agent_response)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response


def evaluating_scenario_agent(agent_response):
    prompt = """
    Evaluate this AI agent response containing a data analysis scenario with a sample dataset against the following strict criteria:

    1. **Scenario Quality**: The scenario description must be concise (2-3 lines), engaging, and clearly explain the context.
    2. **Dataset Accuracy and Relevance**: The provided dataset must include realistic and meaningful column names. The dataset should align with the scenario's context.
    3. **Accuracy**: Ensure that the response is factually correct and free of errors.

    **Task:** Return ONLY "good" if all criteria are met, otherwise "not good".

    **Forced Format:** 
    Final evaluation: [strictly two words]

    Agent's Response:
    {agent_response}
    """

    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.format(agent_response=agent_response)
    response = model.invoke(formatted_prompt, config={"callbacks": [tracer]})
    return response
