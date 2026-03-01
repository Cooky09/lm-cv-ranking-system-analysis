'''
# Run the following commands in your terminal:
!pip install --upgrade pip setuptools wheel
!pip install nltk
!pip install PyMuPDF
!pip install prettytable
!pip install python-docx

# Install cloudmersive api packages
!pip install cloudmersive-virus-api-client
!pip install loguru

# Install pyresparser
!pip uninstall -y spacy
!pip install spacy==2.3.5
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
#!pip install spacy==2.1.0
!pip install pyresparser
!pip install pymupdf
#!python -m spacy download en_core_web_sm
'''

from __future__ import print_function

import spacy
spacy.load('en_core_web_sm')

# Import the necessary packages in your script
import re
import os
import fitz
import time
import nltk
import requests
import logging
import warnings
import numpy as np
import pandas as pd
import cloudmersive_virus_api_client

from docx import Document
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from prettytable import PrettyTable
from nltk.stem import WordNetLemmatizer
#from multiprocessing import Pool, cpu_count
from billiard import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from loguru import logger
from pyresparser import ResumeParser
from io import StringIO

# Packages for Unit Testing
import unittest
from unittest.mock import patch, MagicMock, create_autospec
from io import StringIO

# Include packages related to Cloudmersive
from cloudmersive_virus_api_client.rest import ApiException

# These two functions "scan_one_file" and "scan_all_files_in_repository"
# scan the files to detect potential Virus/Trojan/Malware ... etc. in documents (Sicheng)
def scan_one_file(input_file, api_key, retry_time):
    """
    Scans a single file for viruses and returns whether it is clean.

    Args:
        input_file (str): The path to the file to be scanned.
        api_key (str): The API key for the Cloudmersive virus scan service.
        retry_time (int): The current retry count.

    Returns:
        bool: True if the file is clean,
              False if the file is malicious
              or if an error occurs after reaching retry limit.
    """
    # Configure API key authorization: Apikey
    configuration = cloudmersive_virus_api_client.Configuration()
    configuration.api_key['Apikey'] = api_key

    # Create an instance of the API class
    api_instance = cloudmersive_virus_api_client.ScanApi(
        cloudmersive_virus_api_client.ApiClient(configuration)
    )

    # File type restriction
    restrict_file_types = '.doc,.docx,.pdf'
    allow_executables = False  # No .exe
    allow_invalid_files = False
    allow_scripts = True
    allow_password_protected_files = True
    allow_macros = True
    allow_xml_external_entities = False
    allow_insecure_deserialization = False
    allow_html = False

    try:
        # Send files to Cloudmersive to scan
        api_response = api_instance.scan_file_advanced(
            input_file,
            allow_executables=allow_executables,
            allow_invalid_files=allow_invalid_files,
            allow_scripts=allow_scripts,
            allow_password_protected_files=allow_password_protected_files,
            allow_macros=allow_macros,
            allow_xml_external_entities=allow_xml_external_entities,
            allow_insecure_deserialization=allow_insecure_deserialization,
            allow_html=allow_html
        )

        logger.info(f"File {input_file} scanned by Cloudmersive successfully.")
        return api_response.clean_result  # Return True if the file is clean, False if not

    except Exception as e:
        logger.info(f"Exception when calling Cloudmersive API -> scan_file_advanced: {e}\n")

        if retry_time < 5:
            time.sleep(1)
            return scan_one_file(input_file, api_key, retry_time + 1)
        else:
            logger.warning(f"Failed to scan file: {input_file}")
            return False


def scan_all_files_in_repository(cv_folder_path, all_valid_files, api_key):
    """
    Scans all valid files in a directory and
    returns a list of files without any problems.

    Args:
        cv_folder_path (str): Path to the directory containing the files.
        all_valid_files (list): List of all valid files to scan.
        api_key (str): The API key for the Cloudmersive virus scan service.

    Returns:
        list: A list of filenames that passed the safety check.
    """
    files_without_problem = []

    # check if all files are valid
    # pick only valid ones
    try:
        for file_name in all_valid_files:
            file_path = os.path.join(cv_folder_path, file_name)
            scan_result = scan_one_file(file_path, api_key, 0)

            if scan_result:
                logger.info(f"File {file_path} is clean.")
                files_without_problem.append(file_name)
            else:
                logger.warning(f"File {file_path} is not clean.")

        logger.info("Scanning all files in the repositorycompleted.")
    except Exception as e:
        logger.warning(f"Failed to complete scanning process: {e}")
    finally:
        return files_without_problem

# Function to read the content of DOCX file
def read_docx(file_path):
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Function to read the content of DOC file
def read_doc(file_path):
    result = os.popen(f'antiword "{file_path}"').read()
    return result

# Function to read the content of PDF file
def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Function to check if the directory exists and is accessible
def check_repository_access(cv_folder_path_given):
    try:
        if not os.path.isdir(cv_folder_path_given):
            raise FileNotFoundError(
                "The specified folder does not exist or cannot be accessed."
            )
        return True
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

# Function to list all PDF, DOC, and DOCX files in the CV folder, return [] if no such file exists
def load_cv_files(cv_folder_path_given):
    # I added this 'try ... except' part, only the 'cv_files_found ...' is original
    try:
        if not check_repository_access(cv_folder_path_given):
            # Directory is not accessible, return an empty list
            logger.error("Directory access check failed. No files to load.")
            return []

        cv_files_found = [
            f for f in os.listdir(cv_folder_path_given)
            if f.lower().endswith(('.pdf', '.doc', '.docx'))
        ]

        if cv_files_found:
            logger.info(f"Found {len(cv_files_found)} files: {cv_files_found}")
        else:
            logger.info("No CV files of given format found in the directory.")

        return cv_files_found

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []

# These two functions scan the files to detect potential Virus/Trojan/Malware ... etc. in documents (Sicheng)

def scan_one_file(input_file, api_key, retry_time):
    """
    Scans a single file for viruses and returns whether it is clean.

    Args:
        input_file (str): The path to the file to be scanned.
        api_key (str): The API key for the Cloudmersive virus scan service.
        retry_time (int): The current retry count.

    Returns:
        bool: True if the file is clean,
              False if the file is malicious
              or if an error occurs after reaching retry limit.
    """
    # Configure API key authorization: Apikey
    configuration = cloudmersive_virus_api_client.Configuration()
    configuration.api_key['Apikey'] = api_key

    # Create an instance of the API class
    api_instance = cloudmersive_virus_api_client.ScanApi(
        cloudmersive_virus_api_client.ApiClient(configuration)
    )

    # File type restriction
    restrict_file_types = '.doc,.docx,.pdf'
    allow_executables = False  # No .exe
    allow_invalid_files = False
    allow_scripts = True
    allow_password_protected_files = True
    allow_macros = True
    allow_xml_external_entities = False
    allow_insecure_deserialization = False
    allow_html = False

    try:
        # Send files to Cloudmersive to scan
        api_response = api_instance.scan_file_advanced(
            input_file,
            allow_executables=allow_executables,
            allow_invalid_files=allow_invalid_files,
            allow_scripts=allow_scripts,
            allow_password_protected_files=allow_password_protected_files,
            allow_macros=allow_macros,
            allow_xml_external_entities=allow_xml_external_entities,
            allow_insecure_deserialization=allow_insecure_deserialization,
            allow_html=allow_html
        )

        return api_response.clean_result  # Return True if the file is clean, False if not

    except ApiException as e:
        print(f"Exception when calling ScanApi->scan_file: {e}\n")

        if retry_time < 5:
            time.sleep(1)
            return scan_one_file(input_file, api_key, retry_time + 1)
        else:
            print(f"Fail to scan file {input_file}")
            return False

def scan_all_files_in_repository(cv_folder_path, all_valid_files, api_key):
    """
    Scans all valid files in a directory and
    returns a list of files without any problems.

    Args:
        cv_folder_path (str): Path to the directory containing the files.
        all_valid_files (list): List of all valid files to scan.
        api_key (str): The API key for the Cloudmersive virus scan service.

    Returns:
        list: A list of filenames that passed the safety check.
    """
    files_without_problem = []

    # check if all files are valid
    # pick only valid ones
    for file_name in all_valid_files:
        file_path = os.path.join(cv_folder_path, file_name)
        scan_result = scan_one_file(file_path, api_key, 0)

        if scan_result:
            files_without_problem.append(file_name)

    return files_without_problem

def auto_correct_date(date_str):
    """Attempts to correct common issues with malformed date strings."""
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    # Check if the month part is one character short
    if len(date_str) == 6 and date_str[1] == ' ':
        for month in months:
            if month.startswith(date_str[0]):
                return month + date_str[1:]
    return date_str  # Return the original if no correction was made

def parse_date(date_str):
    """Attempts to parse a date string into a datetime object."""
    date_str = auto_correct_date(date_str)  # Apply auto-correction
    date_formats = [
        "%B %Y", "%m/%Y", "%Y", "%b %Y",  # Handle abbreviations like "Jan 2023"
        "%B %d, %Y", "%b %d, %Y",          # Handle dates like "January 15, 2023"
        "%Y-%m-%d", "%Y/%m/%d",            # Handle ISO formats
        "%m-%Y"                            # Handle "MM-YYYY" format
    ]
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    print(f"Warning: Date '{date_str}' does not match any expected format.")
    return None

def extract_experience_details(text):
    """Extracts experience details from the provided text using regular expressions."""
    experience_pattern = re.compile(
        r'(?P<role>[^\n,]+?)\s*[\n-]*\s*(?P<company>[^\n]+?)\s*[\n-]*\s*(?P<start_date>(\w+\s\d{4})|(\d{2}/\d{4}))\s*[-–to]+\s*(?P<end_date>(\w+\s\d{4})|(\d{2}/\d{4})|Present|Current|Summer \d{4})',
        re.IGNORECASE)
    matches = experience_pattern.finditer(text)
    experience_details = []
    total_months = 0
    start_time = time.time()

    for match in matches:
        if time.time() - start_time > 60:
            print("Processing time exceeded 60 seconds.")
            break

        role = match.group('role').strip()
        company = match.group('company').strip()
        start_date = match.group('start_date').strip()
        end_date = match.group('end_date').strip()

        start = parse_date(start_date)
        if not start:
            continue

        end = datetime.now() if end_date.lower() in ['present', 'current', 'summer'] else parse_date(end_date)
        if not end:
            continue

        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_months += months
        years = round(months / 12, 1)
        experience_details.append(f"{role} at {company} - {years} years")

    return experience_details, total_months

def extract_education(text):
    """Extracts education information from the provided text using regular expressions."""
    pattern = re.compile(r"(?i)(?:Bsc|\bB\.\w+|\bM\.\w+|\bPh\.D\.\w+|\bBachelor(?:'s)?|\bMaster(?:'s)?|\bPh\.D)\s(?:\w+\s)*\w+")

    matches = pattern.findall(text)
    education = ''

    if matches:
        education = ''.join(matches[0]).strip()
    return education

# Lemmatize function
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

# Calculate the similarity between job description and CVs
def calculate_similarity(jd, cv):
    jd_lem = lemmatize(jd)
    cv_lem = lemmatize(cv)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_lem, cv_lem])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Function to read job description files
def read_job_description(file_path):
    text = ""
    if file_path.lower().endswith('.pdf'):
        text = read_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = read_docx(file_path)
    elif file_path.lower().endswith('.doc'):
        text = read_doc(file_path)
    return text

# Function to process CVs in chunks of 100
def process_cvs_in_chunks(cv_files, cv_folder_path, chunk_size=100):
    results = []
    for i in range(0, len(cv_files), chunk_size):
        chunk_files = cv_files[i:i + chunk_size]

        file_paths = [os.path.abspath(os.path.join(cv_folder_path, f)) for f in chunk_files]

        with Pool(cpu_count()) as p:
            chunk_results = p.map(process_resume, file_paths)
        results.extend(chunk_results)
    return results



# function to process single file, return
def process_resume(file_path):
    full_text = ''
    name = ''
    designation = ''
    experience = ''
    education = ''
    skills = ''

    cv_file = os.path.basename(file_path)

    try:
        if not file_path.lower().endswith(('.pdf', '.docx', '.doc')):
            return name, designation, experience, skills, education
        else:
            logger.info(f"Processing file: {file_path}")

            # Determine file type and extract text accordingly
            if file_path.lower().endswith('.pdf'):
                full_text = read_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                full_text = read_docx(file_path)
            elif file_path.lower().endswith('.doc'):
                full_text = read_doc(file_path)

        data = ResumeParser(file_path).get_extracted_data()

        # Extract name, designation, skills, full_experience
        name = data.get('name', 'Unknown')
        education = extract_education(full_text)

        designation = data.get('designation', 'Unknown')
        if designation != 'Unknown' and designation:
            designation = ' '.join(designation)
        elif designation == None:
            designation = ''
        else:
            designation = ''

        skills = data.get('skills', [])
        if skills != []:
            skills = ' '.join(skills)
        else:
            skills = ''

        experience_details, total_months = extract_experience_details(full_text)
        total_years = round(total_months / 12, 1)

        if experience_details:
            experience = ' '.join(experience_details)
        else:
            experience = ''
        total_years =  " Total Years of Experience: " + str(total_years)
        experience = experience + total_years

        # Additional cleanup: Remove bullet points and other unwanted characters
        unwanted_chars = ["•", "●", "▪", "§", "\n", "\r", "○"]
        for char in unwanted_chars:
            name = (name or '').replace(char, "").strip()
            designation = (designation or '').replace(char, "").strip()
            experience = (experience or '').replace(char, "").strip()
            education = (education or '').replace(char, "").strip()
            skills = (skills or '').replace(char, "").strip()

        # Remove 'Email' from the name field if present
        if 'email' in name.lower():
            name = re.split(r'\s+', name, 1)[0]

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

    finally:
        # Print the extracted information
        return cv_file, full_text, name, designation, experience, education, skills

def process_cvs_multiprocess(cv_folder_path):
    # Download NLTK resources if not already downloaded
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    # Check if the repository is accessible
    if not check_repository_access(cv_folder_path):
        logging.warning("Access to the CV repository is not available. Please check the path and your permissions.")
        return [], [], [], [], [], [], []
    
    # List all PDF, DOC, and DOCX files in the CV folder
    cv_files = load_cv_files(cv_folder_path)
    logging.info(f"Loaded CV files: {cv_files}")

    # process cvs in chunks
    cv_results = process_cvs_in_chunks(cv_files, cv_folder_path)

    # decompress the result
    if cv_results:
        file_names, cv_texts, names, designations, experiences, educations, skills_list = zip(*cv_results)
    else:
        file_names, cv_texts, names, designations, experiences, educations, skills_list = [], [], [], [], [], [], []
    
    return file_names, cv_texts, names, designations, experiences, educations, skills_list

def calculate_weighted_similarity(jd, cv_experience, cv_education, cv_skills, weights):
    """
    Calculates the weighted similarity between job description and CV sections.

    Args:
        jd (str): The job description text.
        cv_experience (str): The experience section of the CV.
        cv_education (str): The education section of the CV.
        cv_skills (str): The skills section of the CV.
        weights (dict): Weights assigned to each section.

    Returns:
        float: The weighted similarity score.
    """
    # Calculate similarity for each section
    experience_similarity = calculate_similarity(jd, cv_experience)
    education_similarity = calculate_similarity(jd, cv_education)
    skills_similarity = calculate_similarity(jd, cv_skills)
    
    # Compute weighted sum
    weighted_similarity = (
        weights["Experience"] * experience_similarity +
        weights["Education"] * education_similarity +
        weights["Skills"] * skills_similarity
    )
    
    return weighted_similarity

# Function to calculate similarity scores
def calculate_similarity_scores(cv_texts, jd_text):
    similarity_scores = [calculate_similarity(jd_text, cv_text) for cv_text in cv_texts]
    return similarity_scores

# Function to calculate similarity scores
def calculate_weighted_similarity_scores(jd_text, cv_experience, cv_education, cv_skills):
    """
    Calculates the similarity scores for each CV section based on job description.

    Args:
        cv_texts (list): A list of CV texts (can include summary or overall content).
        cv_experience (list): A list of experiences from the CV.
        cv_education (list): A list of educational qualifications from the CV.
        cv_skills (list): A list of skills from the CV.
        jd_text (str): The job description text.
        section_weights (dict): Weights assigned to each section.

    Returns:
        list: A list of weighted similarity scores.
    """
    similarity_scores = []

    # Define weights for different sections
    section_weights = {
        "Experience": 0.5,  # 50%
        "Education": 0.2,   # 20%
        "Skills": 0.3       # 30%
    }

    # Iterate through each CV section (assuming they have the same length)
    for i in range(len(cv_experience)):
        # Calculate similarity for each section using the previous calculate_weighted_similarity function
        weighted_similarity = calculate_weighted_similarity(
            jd_text,
            cv_experience[i],
            cv_education[i],
            cv_skills[i],
            section_weights
        )
        similarity_scores.append(weighted_similarity)
    
    return similarity_scores

if __name__ == "__main__":
    # Setup logging
    logger.add(
        "application.log",
        rotation="1 week",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name}:{function} | {message}"
    )
    logger.info("Started")

    # Path to the folder containing CVs and job descriptions in Google Drive
    cv_folder_path = 'cv_folder'
    jd_folder_path = 'jd_folder'

    file_names, cv_texts, names, designations, experiences, educations, skills_list = process_cvs_multiprocess(cv_folder_path)

    # check all outputs
    print("-------------------------------------")
    print("file_names: ", file_names)
    print("cv_texts: ", cv_texts)
    print("names: ", names)
    print("designations: ", designations)
    print("experiences: ", experiences)
    print("educations: ", educations)
    print("skills_list: ", skills_list)
    print("-------------------------------------")
    print()

    # Create a DataFrame from the CV texts and extracted information
    cvs_df = pd.DataFrame({
        "File Name": file_names,
        "Text": cv_texts,
        "Name": names,
        "Designation": designations,
        "Experience": experiences,
        "Education": educations,
        "Skills": skills_list
    })

    # Specify the job description file
    job_description_file = 'Python Developer JD.docx'  # Update this to your specific job description file
    jd_path = os.path.join(jd_folder_path, job_description_file)

    # Read the job description
    jd_text = read_job_description(jd_path)
    sample_jd = jd_text  # Use the content of the job description file

    # Calculate weighted similarity scores
    similarity_scores = calculate_weighted_similarity_scores(sample_jd, experiences, educations, skills_list)
    cvs_df['Similarity'] = similarity_scores

    # Check if this is a recommended candidate
    cvs_df['Recommendation'] = cvs_df['Similarity'].apply(lambda x: 'Recommended' if x >= 0.7 else 'Not Recommended')

    # Sort the DataFrame by 'Similarity' in descending order and get the top 10
    cvs_df.sort_values(by='Similarity', ascending=False, inplace=True)

    # Save the results to a CSV file
    output_csv_path = 'output/recommendations.csv'
    cvs_df.to_csv(output_csv_path, index=False)

    # Display the top recommendations
    print("Top Recommendations:")
    print(cvs_df.head(10))

    '''
    # Display the top 10 results in a table format
    results_table = PrettyTable()
    results_table.field_names = ["File Name", "Similarity (%)", "Recommendation"]

    for index, row in top_10_cvs.iterrows():
        similarity_percentage = row['Similarity'] * 100
        results_table.add_row([row['File Name'], f"{similarity_percentage:.2f}%", row['Recommendation']])

    print(f"Top 10 CVs for Job Description: {job_description_file}")
    print(results_table)
    '''

    logger.remove() # Close the logger
