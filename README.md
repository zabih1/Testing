# llm-content-generation

## Brief Overview of Repository
There are three `.py` files
1. `concept.py` : It contains functions:
    - `tech_summary_agent` that generates technical summary of the skill.
    - `fundamental_summary_agent` that generates fundamental (language agnostic) summary using the response from `tech_summary_agent`.
    - `fundamental_quiz_agent` that generates MCQs (question, 4 options, and correct options) using response from `fundamental_summary_agent`.

2. `syntax.py` : It contains functions:
    - `syntactical_quiz_agent` that generates syntax-based MCQs (question, 4 options, and correct options) using the response from `tech_summary_agent`

3. `scenario.py` : It contains functions:
    - `scenario_agent` that generates a real-world scenario based on a field/area of interest. It also outputs a dummy data that is associated with the scenario.
    - `scenario_quiz_agent` that generates MCQs (question, 4 options, and correct options) using response from `scenario_agent` and `tech_summary_agent`.

We use `playground.ipynb` to check the response of the agents.

## Setup Instructions

### 1. Clone the repository

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
```
### 3. Activate the Virtual Environment
```bash
source venv/bin/activate
```
### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

