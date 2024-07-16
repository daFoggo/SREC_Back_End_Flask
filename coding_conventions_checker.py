import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import os
import openai
from langchain_community.document_loaders import PyPDFLoader
import docx
import subprocess

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_error(text):
    check_code = ["E1", "E2", "E3", "E4", "E5", "E7", "E9", "W1", "W2", "W3"]
    error_code = [x for x in text.split() if x[:2] in check_code][0]
    error_description = text.replace(error_code, "").strip()
    return error_code, error_description


def coding_conventions_checker(coding_language, text):
    client = openai.OpenAI()

    def get_completion(prompt):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    coding_conventions: str = ResponseSchema(
        name="issues",
        description="Based on the code snippet, evaluate its coding conventions overall.",
    )
    conversation_metadata_output_schema_parser = (
        StructuredOutputParser.from_response_schemas([coding_conventions])
    )
    conversation_metadata_output_schema = (
        conversation_metadata_output_schema_parser.get_format_instructions()
    )

    conversation_metadata_prompt_template_str = """
    You are someone who evaluates code snippets. 
    I will provide the code snippet, and you need to assess its coding conventions overall. 
    Coding conventions include general comments, any errors present, and whether the code functions correctly.

    << FORMATTING >>
    {format_instructions}

    << INPUT >>
    {chat_history}

    << OUTPUT (remember to include the ```json)>>"""

    conversation_metadata_prompt_template = PromptTemplate.from_template(
        template=conversation_metadata_prompt_template_str
    )

    coding_conventions_checker_result = []

    if coding_language.lower().strip() == "python":
        with open("coding_conventions_checker.py", "w", encoding="utf-8") as file:
            file.write(text)
        result = subprocess.run(
            ["pycodestyle", "coding_conventions_checker.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        maximum_score = 100
        with open(
            "./static/coding_conventions/code_conventions_check.json",
            "r",
            encoding="utf-8",
        ) as file:
            code_conventions_check = json.load(file)
        for error_statement in result.stdout.split("\n"):
            if len(error_statement.split(": ")) == 2:
                error_statement = error_statement.split(": ")[1]
                error_code, error_description = get_error(error_statement)
                try:
                    error_data = code_conventions_check[error_code]
                    error_score = error_data["severity"]
                    print(
                        f"Found {error_code}: {error_description}\n     ==> maximum score: {maximum_score} - {error_score} = {maximum_score - error_score}"
                    )
                    maximum_score -= error_score
                except:
                    print("Not found {error_code} error!")
        os.remove("coding_conventions_checker.py")
    else:
        maximum_score = "Unable to score not-python file"

    messages = [{"role": "user", "content": text}]
    conversation_metadata_recognition_prompt = (
        conversation_metadata_prompt_template.format(
            chat_history=messages,
            format_instructions=conversation_metadata_output_schema,
        )
    )
    conversation_metadata_detected_str = get_completion(
        conversation_metadata_recognition_prompt
    )
    # conversion from string to python dict
    conversation_metadata_detected = conversation_metadata_output_schema_parser.parse(
        conversation_metadata_detected_str
    )
    
    return str(maximum_score), conversation_metadata_detected["issues"]
