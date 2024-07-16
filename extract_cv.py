import os
import docx
import json
import openai
import numpy as np
from extract_job import extract_job
from numpy.linalg import norm
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
def extract_cv(input_directory, job_directory):

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_KEY")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

    job, job_data, categories = extract_job(job_directory)
    job_name = list(job.keys())[0]
    job_description = job[job_name]
    job_description_embedding = embeddings.embed_documents([job_description])
    job_description_embedding = np.array(job_description_embedding[0])

    ranked_candidates = []
    # function to call OpenAI APIs
    def get_completion(prompt):
        messages = [{"role": "user","content": prompt}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content

    name_field = ResponseSchema(name="name", description=f"Based on the candidate's resume document, extract the candidate's name. In a resume, there must be the candidate's name information somewhere in the candidate's resume document (oftens on header, top of the document or in the personal information segment). Please return the answer as a string")
    age_field = ResponseSchema(name="age", description="Based on the candidate's resume document and extract the candidate's age. Please return the answer as a string. Use 'none' if it is not available")
    gmail_field = ResponseSchema(name="gmail", description="Based on the candidate's resume document and extract the candidate's gmail. Gmail often ends with @gmail.com or @email.com. Please return the answer as a string")
    skills_field = ResponseSchema(name="skills", description="Based on the skills segment and project segment of the candidate's resume document, extract names of skills that candidate has. Please return the answer as a string.")
    academic_field = ResponseSchema(name="academic", description="Based on the candidate's resume document, extract type of academic qualification that candidate has. Please return the answer as a string.")
    major_field = ResponseSchema(name="major", description="Based on the academic information, extract the candidate's major field. Please return the answer as a string.")
    language_field = ResponseSchema(name="language", description="Based on the candidate's resume document, extract languages that candidate can use. Please return the answer as a string.")
    certificate_field = ResponseSchema(name="certificate", description="Based on the candidate's resume document, extract names of certificates that candidate has. Please return the answer as a string.")
    experience_field = ResponseSchema(name="experience", description="Based on the experience segment and project segment of the candidate's resume document, extract the candidate's working experience information. The working experience information will have two sub-infors: number of years has been working and working roles. Please return the answer as a string")
    personality_field = ResponseSchema(name="personality", description="Based on the candidate's resume document, extract the candidates's personality traits. User 'none' if it is not available. Please return the answer as a string.")


    # schema with all entities (fields) to be extracted
    conversation_metadata_output_schema_parser = StructuredOutputParser.from_response_schemas(
        [
            name_field,
            age_field,
            gmail_field,
            skills_field,
            academic_field,
            major_field,
            language_field,
            certificate_field,
            experience_field,
            personality_field
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
    candidates = []
    count = 1
    for filename in os.listdir(input_directory):
        text = ''
        filepath = os.path.join(input_directory, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            pages = loader.load_and_split()
            raw_text = ''
            for i in range(len(pages)):
                raw_text += pages[i].page_content
            for x in raw_text.split('\n'):
                text += f'{x.strip().lower()}\n'
        elif filename.endswith('.docx'):
            try: 
                doc = docx.Document(filepath)
                raw_text = []
                for para in doc.paragraphs:
                    raw_text.append(para.text.strip().lower())
                    text = '\n'.join(raw_text)
            except:
                print(f'skip file {count}')
                count += 1
                continue
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

        candidate_data_for_return = {}
        candidate_data_string = []
        for category in conversation_metadata_detected.keys():
            segment = conversation_metadata_detected[category]
            if segment == '':
                segment = 'none'
            candidate_data_for_return.update({category: segment})
            if category not in ('name', 'gmail', 'age'):
                if segment != 'none':
                    candidate_data_string.append(f'{category}: {segment}')

        candidate_data_string = '; '.join(candidate_data_string)
        candidate_data_string_embedding = embeddings.embed_documents([candidate_data_string])
        candidate_data_string_embedding = np.array(candidate_data_string_embedding[0])
        cosine_sim = np.dot(job_description_embedding, candidate_data_string_embedding) / (norm(job_description_embedding) * norm(candidate_data_string_embedding))
        candidate_data_for_return.update({'cv_matching': cosine_sim})
        candidates.append(candidate_data_for_return)
    return candidates, categories, job_data