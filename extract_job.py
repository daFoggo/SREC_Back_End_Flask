import os
import docx
import json
import openai
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

def extract_job(directory):
    client = openai.OpenAI()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_KEY")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # function to call OpenAI APIs
    def get_completion(prompt):
        messages = [{"role": "user","content": prompt}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content

    name_field = ResponseSchema(name="name", description=f"Based on the job's description document, extract the job's name. Please return the answer as a string.")
    age_field = ResponseSchema(name="age", description="Based on the job's description document and extract job's age requirement. Please return the answer as a string. Use 'none' if it is not available")
    experience_field = ResponseSchema(name="experience", description="Based on the experience requirement segment of the job's description document, extract all the job's working experience required. The working experience information will have two sub-infors: number of years has been working and working roles. Please return the answer as a string.")
    skills_field = ResponseSchema(name="skills", description="Based on the skills requirement segment of the job's description document (also remind to check the experience information for full skills requirements), extract all names of skills that job requires candidate to certainly have for the need of the job. Skills include both soft skills such as analytical thinking, logical thinking, problems solving, communication, etc and technical skills such as python, backend, fronend, css, c++, etc. Please return the answer as a string.")
    academic_field = ResponseSchema(name="academic", description="Based on the job's description document, extract type of academic qualification that job requires candidate to have.")
    major_field = ResponseSchema(name="major", description="Based on the academic information, extract job's major requirement. Please return the answer as a string.")
    language_field = ResponseSchema(name="language", description="Based on the job's description document, extract all languages that job requires candidate to be able to use. Please return the answer as a string.")
    certificate_field = ResponseSchema(name="certificate", description="Based on the job's description document, extract all names of certificates that job requires candidate to certainly have for the need of the job. Please return the answer as a string.")
    personality_field = ResponseSchema(name="personality", description="Based on the job's description document, extract all the job's preferred personality traits. Use 'none' if it is not available. Please return the answer as a string.")

    preferred_skills_field = ResponseSchema(name="preferred_skills", description="Based on the job's description document, extract all the job's skills that highly valued but not required. Use 'none' if it is not available. Preferred skills can be writen in the 'highly valued but not required' segment or in the skills information of the job description document. Please return the answer as a string.")
    preferred_certificate_field = ResponseSchema(name="preferred_certificate", description="Based on the job's description document, extract all certificates that highly valued but not required. Use 'none' if it is not available. Preferred certificates can be writen in the 'highly valued but not required' segment or in the skills information of the job description document. Please return the answer as a string.")

    # schema with all entities (fields) to be extracted
    conversation_metadata_output_schema_parser = StructuredOutputParser.from_response_schemas(
        [
            name_field,
            age_field,
            skills_field,
            academic_field,
            major_field,
            language_field,
            certificate_field,
            experience_field,
            personality_field,
            preferred_skills_field,
            preferred_certificate_field
        ]
    )
    conversation_metadata_output_schema = conversation_metadata_output_schema_parser.get_format_instructions()

    conversation_metadata_prompt_template_str = """
    Given in input text document of a candidate's resume, \
    extract the following metadata according to the format instructions below.
    
    << FORMATTING >>
    {format_instructions}
    
    << INPUT >>
    {chat_history}
    
    << OUTPUT (remember to include the ```json)>>"""
    
    conversation_metadata_prompt_template = PromptTemplate.from_template(template=conversation_metadata_prompt_template_str)

    loader = TextLoader(directory)
    pages = loader.load_and_split()
    name_query = pages[0].page_content
    raw_texts = []
    for raw_text in name_query.split('\n'):
        raw_texts.append(raw_text.strip().lower())
    text = '\n'.join(raw_texts)

    messages =  [{"role": "user", "content": text}]
    # init prompt
    conversation_metadata_recognition_prompt = (
        conversation_metadata_prompt_template.format(
            chat_history=messages,
            format_instructions=conversation_metadata_output_schema
        )
    )
    # call openAI API to detect the conversation metadata (e.g. intent, user_need, entities, etc.)
    conversation_metadata_detected_str = get_completion(conversation_metadata_recognition_prompt)
    # conversion from string to python dict
    conversation_metadata_detected = conversation_metadata_output_schema_parser.parse(conversation_metadata_detected_str)

    categories = []
    job = {}
    job_data = {}
    job_data_string = []
    for category in conversation_metadata_detected.keys():
        if category == 'name':
            continue
        else:
            segment = conversation_metadata_detected[category]
            if segment != 'none' and segment != '':
                categories += [category]
                job_data.update({category: segment})
                job_data_string.append(f'{category}: {segment}')
    job.update({conversation_metadata_detected['name']: job_data})
    # with open('./static/job_descriptions/job_description.json', 'w', encoding = 'utf-8') as f:
    #     json.dump(job, f, ensure_ascii= True)
    job_data_string = '; '.join(job_data_string)
    return {conversation_metadata_detected['name']: job_data_string}, job_data, categories