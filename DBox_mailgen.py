import streamlit as st
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

st.set_page_config(layout="wide")

e_mail = ""
action_points = ""
extra_info = ""

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_and_translate_email(e_mail, llm_model):

    llm = ChatOpenAI(temperature=0.1, model=llm_model)
    
    template_language_detection = """
    Detect the language in which the email between triple backticks is written:
    '''
    email = {e_mail}
    '''
    """
    
    prompt_language_detection = ChatPromptTemplate.from_template(template_language_detection)
    
    chain_language_detection = LLMChain(
        llm=llm, 
        prompt=prompt_language_detection, 
        output_key="Email_language"
    )
    
    template_translate_email = """
    Translate the email between backticks to French if {Email_language} is Dutch and to Dutch if {Email_language} is French.
    If {Email_language} is neither French nor Dutch, translate it to French and Dutch.
    '''
    email = {e_mail}
    '''
    """
    
    prompt_translate_email = ChatPromptTemplate.from_template(template_translate_email)
    
    chain_translate_email = LLMChain(
        llm=llm, 
        prompt=prompt_translate_email, 
        output_key="Email_translation"
    )
    
    template_extract_action_points = """
    Look at the email between triple backticks and extract all action points from it: 
    '''
    email = {e_mail}
    '''
    Consider that the mail is sent by a donor to an NGO. The action points to be listed are only those for the NGO to take care of. 
    List the action points and add a translation in French if {Email_language} is Dutch and in Dutch if {Email_language} is French. 
    
    The list should have the format as in the following example between triple backticks:
    
    Example:
    '''
    1. Mettre fin au mandat dans les 24 heures / het mandaat stopzetten binnen de 24 uur
    2. Confirmer par mail quand c'est fait / Per mail bevestigen wanneer het is stopgezet
    '''
    """
    
    prompt_extract_action_points = ChatPromptTemplate.from_template(template_extract_action_points)
    
    chain_extract_action_points = LLMChain(
        llm=llm, 
        prompt=prompt_extract_action_points, 
        output_key="Email_action_points"
    )
      
    overall_chain = SequentialChain(
        chains=[chain_language_detection, chain_translate_email, chain_extract_action_points],
        input_variables=['e_mail'],
        output_variables=["Email_language", "Email_translation", "Email_action_points"],
        verbose=False
    )
    
    # Invoke the overall chain
    result = overall_chain({
        'e_mail': e_mail
    })

    return result

def reply_to_email(e_mail, done_action_points, extra_info, llm_model):

    llm = ChatOpenAI(temperature=0.1, model=llm_model)
    
    template_language_detection = """
    Detect the language in which the email between triple backticks is written:
    '''
    email = {e_mail}
    '''
    """
    
    prompt_language_detection = ChatPromptTemplate.from_template(template_language_detection)
    
    chain_language_detection = LLMChain(
        llm=llm, 
        prompt=prompt_language_detection, 
        output_key="Email_language"
    )
       
    template_extract_action_points = """
    Look at the email between triple backticks and extract all action points from it: 
    '''
    email = {e_mail}
    '''
    Consider that the mail is sent by a donor to an NGO. The action points to be listed are only those for the NGO to take care of. 
    List the action points and add a translation in French if {Email_language} is Dutch and in Dutch if {Email_language} is French. 
    
    The list should have the format as in the following example between triple backticks:
    
    Example:
    '''
    1. Mettre fin au mandat dans les 24 heures / het mandaat stopzetten binnen de 24 uur
    2. Confirmer par mail quand c'est fait / Per mail bevestigen wanneer het is stopgezet
    '''
    """
    
    prompt_extract_action_points = ChatPromptTemplate.from_template(template_extract_action_points)
    
    chain_extract_action_points = LLMChain(
        llm=llm, 
        prompt=prompt_extract_action_points, 
        output_key="Email_action_points"
    )
    
    template_propose_answer = """
    Your task is to propose an answer in {Email_language} to the email between triple backticks:
    '''
    email = {e_mail}
    '''
    Consider the {Email_action_points} and mention, if needed, that the following action points between triple backticks have been taken care of:
    '''
    checked action points = {done_action_points}
    '''
    Consider also the following info between backticks:
    '''
    info = {extra_info}
    '''
    Your answer should always be engaging, constructive, helpful and respectful. Consider not only the content but also the tone and sentiment of the donor message to determine the most suitable answer. 
    Avoid any kind of controversy, ambiguities, or politically oriented answers.
    If appropriate, while avoiding being too pushy or inpolite, mention the possibility to become a (regular) donor (again) by surfing to our website www.medecinsdumonde.be or www.doktersvandewereld.be (according to {Email_language}). 
    Try to end by a positive note and/or a thank you.
    """
    
    prompt_propose_answer = ChatPromptTemplate.from_template(template_propose_answer)
    
    chain_propose_answer = LLMChain(
        llm=llm, 
        prompt=prompt_propose_answer, 
        output_key="Email_answer"
    )
    
    template_translate_answer = """
    Translate the email answer to French if {Email_language} is Dutch and to Dutch if {Email_language} is French :
    '''
    email answer = {Email_answer}
    '''
    If {Email_language} is neither French nor Dutch, translate it to both French and Dutch.
    """
    
    prompt_translate_answer = ChatPromptTemplate.from_template(template_translate_answer)
    
    chain_translate_answer = LLMChain(
        llm=llm, 
        prompt=prompt_translate_answer, 
        output_key="Email_answer_translation"
    )
    
    overall_chain = SequentialChain(
        chains=[chain_language_detection, chain_extract_action_points, chain_propose_answer, chain_translate_answer],
        input_variables=['e_mail', 'done_action_points', 'extra_info'],
        output_variables=["Email_language", "Email_action_points", "Email_answer", "Email_answer_translation"],
        verbose=False
    )
    
    # Invoke the overall chain
    result = overall_chain({
        'e_mail': e_mail,
        'done_action_points': done_action_points,
        'extra_info': extra_info
    })

    return result

def main():
    PASSWORD = st.secrets["MDM_PASSWORD"]
    client = OpenAI()
    
    pass_word = st.sidebar.text_input('Enter the password:')
    if not pass_word:
        st.stop()
    if pass_word != PASSWORD:
        st.error('The password you entered is incorrect.')
        st.stop()

    selected_model = st.sidebar.radio('**Select your MODEL:**', ['gpt-4o', 'gpt-4-turbo'])

    st.subheader("Donorsbox Reply Tool")
    
    st.write("paste the email here for which you would like ChatGPT to generate a response.")
    st.write("**Remove all personal information from the email.**")
    e_mail = st.text_area('Paste email', height=150)

    result_1 = extract_and_translate_email(e_mail, selected_model)

    if st.button("Click here to translate the original email and extract action points"):
        st.write("*Translation*")
        st.write(result_1['Email_translation'])
        st.write("*Action points*")
        st.write(result_1['Email_action_points'])
        
    col1, col2 = st.columns(2)
    with col1: 
        st.write("Paste here the action points you have or will have completed by the time you will answer the mail.")
        action_points = st.text_area('Mention action points', height=150)
    with col2: 
        st.write("Paste additional information you want to see mentionned in the answer, and which is not an action point.")
        extra_info = st.text_area('Add extra info', height=150)

    result_2 = reply_to_email(e_mail, action_points, extra_info, selected_model)
    if st.button("Click here to generate draft answer"):
        st.write('*Proposed answer to the mail*')
        st.write(result_2['Email_answer'])
        st.write('*Translation of answer*')
        st.write(result_2['Email_answer_translation'])


if __name__ == "__main__":
    main()

