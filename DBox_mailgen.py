import streamlit as st
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

st.set_page_config(layout="wide")

name_input = ""
e_mail = ""
action_points = ""
extra_info = ""

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_and_translate_email(e_mail, llm_model):

    llm = ChatOpenAI(temperature=0.1, model=llm_model)
      
    template_translate_email = """
    Translate the email between backticks a) to French if it is written in Dutch and b) to Dutch if it is written in French.
    If it is written in any other language, translate it c) to both French and Dutch.
    '''
    {e_mail}
    '''
    Render only the translation(s), without any introduction or comment.
    """
    
    prompt_translate_email = ChatPromptTemplate.from_template(template_translate_email)
    
    chain_translate_email = LLMChain(
        llm=llm, 
        prompt=prompt_translate_email, 
        output_key="Email_translation"
    )
    
    template_extract_action_points = """
    Look at the email between triple backticks and extract every possible action point from it: 
    '''
    {e_mail}
    '''
    Consider that the mail is sent by a donor to an NGO. The action points to be listed are only those for the NGO to take care of. 
    List the action points in both Dutch and French. 
    
    The list should have the format as follows:
    
    Format example:
    1. Mettre fin au mandat dans les 24 heures / het mandaat stopzetten binnen de 24 uur
    2. Confirmer par mail quand c'est fait / Per mail bevestigen wanneer het is stopgezet
    """
    
    prompt_extract_action_points = ChatPromptTemplate.from_template(template_extract_action_points)
    
    chain_extract_action_points = LLMChain(
        llm=llm, 
        prompt=prompt_extract_action_points, 
        output_key="Email_action_points"
    )
      
    overall_chain = SequentialChain(
        chains=[chain_translate_email, chain_extract_action_points],
        input_variables=['e_mail'],
        output_variables=["Email_translation", "Email_action_points"],
        verbose=False
    )
    
    # Invoke the overall chain
    result = overall_chain({
        'e_mail': e_mail
    })

    return result

def reply_to_email(e_mail, name, done_action_points, extra_info, temperature, llm_model):

    llm = ChatOpenAI(temperature=temperature, model=llm_model)
            
    template_propose_answer = """
    Your task is to draft a response to the email enclosed within triple backticks, in the same language:
    ```
    {e_mail}
    ```
    Mention that the following action points have been addressed if appropriate:
    ```
    {done_action_points}
    ```
    Consider the additional information below and use it if adequate:
    ```
    {extra_info}
    ```
    Your response should be engaging, constructive, helpful, and respectful. Reflect on the tone and sentiment of the sender's message to determine the most suitable reply. Avoid controversy, ambiguity, or politically oriented responses.

    If the sender requested to stop or cancel a regular donation, politely mention the option to become a regular donor again by visiting our website:
    - French: www.medecinsdumonde.be
    - Dutch: www.doktersvandewereld.be

    Conclude with a positive note and/or a thank you. Sign off with the appropriate salutations, your name, and the organization:
    - French response: Médecins du Monde
    - Dutch response: Dokters van de Wereld

    Regards,
    {name}
    """
    
    prompt_propose_answer = ChatPromptTemplate.from_template(template_propose_answer)
    
    chain_propose_answer = LLMChain(
        llm=llm, 
        prompt=prompt_propose_answer, 
        output_key="Email_answer"
    )
    
    template_translate_answer = """
    Translate the email answer between tripple backticks a)into French if the email answer is in Dutch or b) into Dutch if the email answer is in French.
    If the email answer is nor in French nor in Dutch, translate it to c) both French and Dutch.
    
    '''
    {Email_answer}
    '''
    Render only the translation, without any comment or introduction.
    """
    
    prompt_translate_answer = ChatPromptTemplate.from_template(template_translate_answer)
    
    chain_translate_answer = LLMChain(
        llm=llm, 
        prompt=prompt_translate_answer, 
        output_key="Email_answer_translation"
    )
    
    overall_chain = SequentialChain(
        chains=[chain_propose_answer, chain_translate_answer],
        input_variables=['e_mail', 'name', 'done_action_points', 'extra_info'],
        output_variables=["Email_answer", "Email_answer_translation"],
        verbose=False
    )
    
    # Invoke the overall chain
    result = overall_chain({
        'e_mail': e_mail,
        'name': name,
        'done_action_points': done_action_points,
        'extra_info': extra_info
    })

    return result

def main():
    PASSWORD = st.secrets["MDM_PASSWORD"]
    client = OpenAI()
    
    pass_word = st.sidebar.text_input('**Enter the password:**')
    if not pass_word:
        st.stop()
    if pass_word != PASSWORD:
        st.error('The password you entered is incorrect.')
        st.stop()

    selected_model = st.sidebar.radio('**Select your MODEL:**', ['gpt-4o', 'gpt-4-turbo'])
    set_temperature = st.sidebar.slider('**Select the TEMPERATURE**', min_value=0.1, max_value=0.3, step=0.1) 
    
    st.title("Donorsbox Reply Tool")
    
    st.write("Paste the email here for which you want ChatGPT to generate a response.")
    st.write("**Remove all GDPR sensitive information.**")
    e_mail = st.text_area('Paste email', height=150)

    if st.button("Click here to translate the original email and extract action points"):
        result_1 = extract_and_translate_email(e_mail, selected_model)
        st.write("**Translation**")
        st.write(result_1['Email_translation'])
        st.write("**Action points**")
        st.write(result_1['Email_action_points'])
        st.markdown('---')
    
    name_input = st.text_area('Enter your full name', height=5)
    col1, col2 = st.columns(2)
    with col1: 
        st.write("List the action points you have completed or will complete by the time you reply to the email.")
        action_points = st.text_area('Mention action points', height=150)
    with col2: 
        st.write("If any, include additional information to be mentioned in the answer and specify any message to be avoided.")
        extra_info = st.text_area('Add extra info', height=150)

    if st.button("Click here to generate an answer"):
        result_2 = reply_to_email(e_mail, name_input, action_points, extra_info, set_temperature, selected_model)
        st.write('**Proposed answer to the mail**')
        st.write(result_2['Email_answer'])
        st.write('**Translation of answer**')
        st.write(result_2['Email_answer_translation'])


if __name__ == "__main__":
    main()

