import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from datetime import datetime
from langdetect import detect
import openai

st.set_page_config(layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# AI agent functions
def detect_demands(email_content):
    system_template = "You are an AI assistant specialized in analyzing emails and identifying the main demands or requests made by the sender."
    human_template = """
    Analyze the following email and identify the main demands or requests made by the sender. 
    Categorize them into one or more of these types: data change, tax certificate, donation adjustment, complaint, donation cancellation, unsubscribe, or general inquiry.
    If multiple demands are present, list them all.

    Email content:
    {email_content}

    Detected demands (comma-separated list):
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    demands = chain.run(email_content=email_content).strip().split(', ')
    return demands

def select_relevant_responses(demands, examples, email_content, donor_info, actions, additional_messages, additional_guidelines):
    template = """
    Given the following information, search through the provided examples to get inspiration. 
    
    Email content:
    {email_content}

    Detected demands:
    {demands}

    Donor Information:
    {donor_info}

    Actions taken:
    {actions}

    Additional messages to include:
    {additional_messages}

    Additional guidelines to follow:
    {additional_guidelines}
    
    Examples:
    {examples}

    Based on the examples, suggest pertinent response parts, focusing on addressing the detected demands 
    and considering the donor information, actions taken, additional messages and additional guidelines.

    Please format your response as follows:
    1. List the top 3-5 most relevant response parts in order of importance. 
    2. For each part, provide a brief explanation of why it's relevant.
    3. If no exact matches are found in the examples, suggest appropriate response parts based on the given context.
    4. If absolutely no relevant information is found, state this clearly and suggest a general approach for responding.

    Your response:
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are an AI assistant specialized in selecting or suggesting relevant response parts for email inquiries based on various contextual factors. Your goal is to provide the most appropriate and helpful response elements."),
        HumanMessagePromptTemplate.from_template(template)
    ])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    chain = LLMChain(llm=llm, prompt=prompt)

    relevant_responses = chain.run(email_content=email_content, demands=demands, donor_info=donor_info, 
                                   actions=actions, additional_messages=additional_messages, examples=examples, additional_guidelines=additional_guidelines)
    return relevant_responses


def draft_initial_response(email_content, actions, additional_messages, additional_guidelines, donor_info, relevant_responses, language, name, organization):
    template = """
    Based on the following information, draft an engaging and respectful email response in {language}:
    
    Original Email: {email_content}
    Actions Taken: {actions}
    Additional Messages to Include: {additional_messages}
    Additional Guidelines to Follow: {additional_guidelines}
    Donor Information: {donor_info}
    Relevant Response Parts: {relevant_responses}
    Your Name: {name}
    Organization: {organization}
    
    Guidelines for drafting the response:
    1. Draft a response that is engaging, constructive, helpful, and respectful.
    2. Draw inspiration from the relevant response parts provided but NEVER just literally translate them. Capture the ideas and draft them in {language} as if you would think of them from scratch in {language}.
    3. Utilize the detailed donor information to personalize the response appropriately.
    4. Follow the additional guidelines.
    5. Avoid controversy, ambiguity, or politically oriented responses. Maintain a positive tone throughout the email.
    6. Address all the demands detected in the original email.
    7. Include the additional messages as appropriate within the context of the response.
    8. Conclude with a positive note and/or a thank you.
    9. Use appropriate salutations and sign-off for the email, that are known and commonly used in {language}.
    10. Include the provided name and organization in the signature.
    
    Complete response:
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(f"You are an AI assistant specialized in drafting email responses in {language}, with a focus on donor communication."),
        HumanMessagePromptTemplate.from_template(template)
    ])
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run(email_content=email_content, actions=actions, additional_messages=additional_messages,
                     donor_info=donor_info, relevant_responses=relevant_responses, language=language,
                     name=name, organization=organization, additional_guidelines=additional_guidelines)


def refine_response(draft_response, donor_info, language, name, organization, temperature):
    system_template = f"You are an expert fundraiser specialized in refining email responses in {language}, with a deep understanding of donor psychology and effective communication strategies."
    human_template = """
    Refine the following email draft:
    
    Draft: {draft_response}
    Donor Information: {donor_info}
    Your Name: {name}
    Organization: {organization}
    
    Guidelines for refinement:
    1. Preserve all content elements and messages conveyed in the original draft. Also, DO NOT ADD extra information.
    2. Adapt the phrasing and structure to maximize positive effects on the reader (donor).
    3. Enhance the fluidity and authenticity of the language used, ensuring the email sounds completely natural and seamless for a native speaker of {language}. 
    4. Optimize the email's impact by improving its overall coherence and persuasiveness. Eliminate unnecessary repetitions, and ensure the message is concise.
    5. Ensure the tone remains appropriate for the donor type and situation, while being direct and to-the-point.
    6. Make sure the signature includes the provided name and organization.

    Your task is to refine the form and style of the email while keeping its core content intact. 
    The goal is to make the email more engaging, impactful, and donor-centric without altering its fundamental message or omitting any important information.
    
    Complete refined response:
    """
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=temperature)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
        
    return chain.run(draft_response=draft_response, donor_info=donor_info, language=language, name=name, organization=organization)

def translate_email(email_content, source_language, target_language):
    system_template = f"You are a professional translator specializing in translating from {source_language} to {target_language}."
    human_template = """
    Translate the following email:
    
    {email_content}
    
    Ensure that the translation maintains the tone and intent of the original message and is perfectly fluent.
    """
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    return chain.run(email_content=email_content)

# Example data for replies
examples = """

### EXAMPLES OF DATA CHANGE REQUESTS

***Response***
Dear [Name],

Please email us your new bank account number. We will update our system to ensure your donations continue from the new account. 

Thank you for your invaluable support. Your contributions make a significant difference for those in need. 

Feel free to reach out with any further questions.

With solidarity,
[Your Name]

---
***Response***

Dear [firstname],

Thank you for your valuable support since [date]! We have updated your address in our database, and you will now receive our communications at your new address.

Thank you for standing by our side!

Warm regards,
[Your Name]

---
***Response***

Dear [name],

Thank you for informing us about your address change. We have updated it in our system, and you will now receive our communications at the new address.

Thank you for your loyal support. It is invaluable in helping the most vulnerable.

Take care, and feel free to contact us with any further questions.

Kind regards,
[Your Name]

---
***Response***

Dear [name],

Thank you for your email. We have updated your bank account number in our database. Future direct debits will be processed from the new account.

Thank you for your loyal support. It allows us to provide urgent care to those in need.

If you have any further questions, please let us know.

Kind regards,
[Your Name]

### EXAMPLES OF TAX CERTIFICATE REQUESTS

***Response***

Dear [name],

Tax certificates are automatically sent at the end of February or beginning of March. Yours will be delivered digitally, so please check your spam folder as well. 

If you haven't received it by March 15, please let us know.

Thank you for your loyal support. It significantly helps us assist the most vulnerable.

Kind regards,
[Your Name]

---
***Response***

Dear [firstname],

We have sent you a tax certificate digitally, but it might have ended up in your spam folder. Attached, you will find a duplicate of your tax certificate.

Please contact us again if you need further assistance. We are here to help.

Thank you for your support, which enables us to help vulnerable people.

With solidarity,
[Your Name]

---
***Response***

Dear [name],

Thank you for your donation at [date]. Your support helps us continue our work with vulnerable people.

The tax certificate will be automatically emailed to you between February and March.

If you have any further questions, please contact us. We are happy to assist you.

Kind regards,
[Your Name]

### EXAMPLES OF DONATION ADJUSTMENTS

***Response***

Dear [name],

As requested, we have reduced your monthly donation from €[amount] to €[amount]. We appreciate your continued support, even at a reduced amount. 

If you have any further questions, please let us know. 

Take care of yourself and those who are close to you.

With solidarity,
[Your Name]

### EXAMPLES OF DONATION CANCELLATION

***Response***

Dear Ms. [Name],

Thank you for your email.

As requested, we have canceled your authorization, ensuring that no automatic payments will be processed.

While we respect and understand your decision, we deeply regret losing your support so soon. Your contribution is vital to our mission, enabling us to develop projects that support vulnerable communities.

Should you have any further questions or require assistance, please do not hesitate to reach out. We are always here to help.

We also hope you continue to follow our work through social media or our newsletters. Should you wish to become a donor again, whether through a one-time or monthly contribution, you can easily do so on our website at www.doktersvandewereld.be [IF LANGUAGE == "nl"] / www.medecinsdumonde.be [IF LANGUAGE == "fr"].

Kind regards,

---
***Response***

Dear Ms. [Name],

Thank you for your email.

We have received your request and promptly canceled your authorization, ensuring no automatic payments will occur.

We understand your concerns and apologize if our communication was unclear. Your feedback is invaluable and helps us improve our interactions with future donors.

If you have any further questions, please feel free to contact us. We are here to assist you.

We hope you continue to follow our work through social media or our newsletters.

Thank you again for your understanding.

Best regards,

---
***Response***

Dear [name],

Thank you for your email.

As requested, we have canceled your authorization, ensuring no payments will be processed. We understand and respect your decision and appreciate your initial interest in supporting our organization.

If you have any further questions, please do not hesitate to contact us. We are always here to assist you.

We hope you continue to follow our work through social media or our newsletter.

Wishing you a pleasant day.

With solidarity,

---
***Response***

Dear [name],

Thank you for your email. 

We have canceled your direct debit. There will be no more automatic payments.

Thank you for your past support  It has been crucial in helping the most vulnerable. [ONLY IF DONOR INFORMATION SAYS PREVIOUS GIFTS HAVE BEEN MADE]

Take care and have a nice day.

Warm regards,
[Your Name]

---
***Response***

Dear [name],

We have received your request to cancel your direct debit. As requested, it has been canceled, and there will be no automatic payments.

We understand your decision but are sorry to lose your support so soon. Every bit of support helps us develop projects and assist vulnerable people.

If you are still interested in our work, you can visit our website for more information or to make a one-time donation.

If you have any further questions, please contact us. We are happy to assist you.

Kind regards,
[Your Name]

---
***Response***

Dear [firstname],

Thank you for your message and feedback. We apologize for any miscommunication. Indeed, it concerns monthly donations. We will address this with our field recruiters to prevent future misunderstandings.

We have canceled your direct debit, and no payments will be initiated.

If you have any additional questions, please contact us. We are here to help.

Have a nice day.

Kind regards,
[Your Name]

---
***Response***

Dear [name],

Thank you for your email. As requested, we have canceled your direct debit, and there will be no more automatic payments.

We sincerely thank you for your long and loyal support. It has made a significant difference for those without access to care.

We hope you will continue to follow us on social media or other channels. Making a one-time donation or becoming a donor again is easy via our website.

If you have any further questions, please contact us. We are happy to assist you.

Have a nice day.

Best regards,
[Your Name]

---
***Response***

Dear [Name],

Thank you for your email. We understand your decision and have canceled your mandate, so no automatic payments will be made. 

It's great that you support multiple NGOs. If you ever consider adding us to your list, you know where to find us: www.doktersvandewereld.be [IF LANGUAGE == "nl"] or www.medecinsdumonde.be [IF LANGUAGE == "fr"]

You can follow our projects on social media or through our monthly newsletter.

Have a nice day.

Kind regards,
[Your Name]

### EXAMPLES OF UNSUBSCRIBING FROM COMMUNICATION

***Response***

Dear [name],

Thank you for your email and for sharing your thoughts. We understand your feelings and appreciate your feedback.

Please know that we are very grateful for your support. It is thanks to donors like you that we can continue our work.

We have updated our system to stop sending you emails about our actions or donation requests.

If you have any further questions, please contact us. We are here to ensure our cooperation meets your expectations.

Best regards,
[Your Name]

---
***Response***

Dear [firstname],

Thank you for your support! Your communication preference has been updated. You will now receive only emails from us.

Thank you for standing by our side!

Warm regards,
[Your Name]

---
***Response***

Dear Klaas,

Thank you for your message. We have updated our database, and you will no longer receive postal mail from us. We will send further communications via email.

If you have specific preferences, please let us know.

Have a nice day.

Kind regards,
[Your Name]
"""

# Streamlit app
def main():

    PASSWORD = st.secrets["MDM_PASSWORD"]
        
    pass_word = st.sidebar.text_input('**Enter the password:**')
    if not pass_word:
        st.stop()
    if pass_word != PASSWORD:
        st.error('The password you entered is incorrect.')
        st.stop()
    st.sidebar.write("")
    st.sidebar.write("**IMPORTANT!**")
    st.sidebar.write("Please contact me (Alexis 😊) if the responses frequently do not meet your needs. I may be able to address this by feeding the program with sample responses tailored to your specific use cases.")  
    st.sidebar.write("In general, please provide feedback on any areas or scenarios where the tool seems to fall short.")
    
    st.title("Multiagent AI Email System")

    # Initialize session state
    if 'generated_response' not in st.session_state:
        st.session_state.generated_response = None
    if 'translated_response' not in st.session_state:
        st.session_state.translated_response = None
    if 'adapted_response' not in st.session_state:
        st.session_state.adapted_response = None
    if 'detected_language' not in st.session_state:
        st.session_state.detected_language = None
    if 'target_language' not in st.session_state:
        st.session_state.target_language = None
    if 'translated_original_mail' not in st.session_state:
        st.session_state.translated_original_mail = None

    # Input email
    email_content = st.text_area("Paste the incoming email here:")

    # Detect language
    if email_content:
        st.session_state.detected_language = detect(email_content)
        st.write(f"Detected language: {st.session_state.detected_language}")

        # Set target language based on detected language
        st.session_state.target_language = "Dutch" if st.session_state.detected_language == "fr" else "French"

        # Detect demands
        with st.spinner("Analyzing email..."):
            demands = detect_demands(email_content)

        # st.subheader("Detected Demands")
        # st.write(demands)

    # Translation option
    if st.button("Translate the original email"):
        st.session_state.translated_original_mail = translate_email(email_content, st.session_state.detected_language, st.session_state.target_language)

    # Display translated response
    if st.session_state.translated_original_mail:
        st.subheader(f"Translated Email ({st.session_state.target_language})")
        st.write(st.session_state.translated_response)
    
    # Actions taken and additional messages
    st.subheader("Manage content")
    actions = st.text_area("Specify actions undertaken (eg stop sdd):")
    st.write("**Optional:**")
    col1, col2 = st.columns(2)
    with col1:
        additional_messages = st.text_area("Additional messages (eg apologize for confusion):")
    with col2:
        additional_guidelines = st.text_area("Additional guidelines (eg direct tone):")

    # Donor type selection
    donor_type = st.radio("Select donor type:", 
                          ["Newly recruited regular donor before first donation (or selection)",
                           "Newly recruited regular donor with low number of donations (eg 1 to 4)",
                           "Regular donor with track record", 
                           "Non-regular giver", 
                           "Not a giver"])

    # Additional inputs based on donor type
    donor_info = {"type": donor_type}
    if donor_type == "Newly recruited regular donor before first donation (or selection)":
        donor_info["gift_history_info"] = "Cancelation has been processed, THERE WILL BE NO PAYMENT. NEVER say: there will be no FURTHER payments."
    if donor_type == "Newly recruited regular donor with low number of donations (eg 1 to 4)":
        donor_info["gift_history_info"] = st.text_input("Number of gifts made to mention in reply:")
    if donor_type == "Regular donor with track record":
        col1, col2 = st.columns(2)
        with col1:
            donor_info["gift_history_info"] = st.text_input("Start date of regular gifts or total number of regular gifts:")
        with col2:
            donor_info["regular_gift_amount"] = st.text_input("Amount of regular gift:")
    elif donor_type == "Non-regular giver":
        col1, col2 = st.columns(2)
        with col1:
            donor_info["last_gift_date"] = st.text_input("Date of last gift:")
        with col2:
            donor_info["last_gift_amount"] = st.text_input("Amount of last gift:")

    # Signature
    st.subheader("Email Signature")
    name = st.text_input("Your Name:")
    organization = "Dokters van de Wereld" if st.session_state.detected_language == "nl" else "Médecins du Monde"
    set_temperature = st.slider('**Select the TEMPERATURE of the latest AI agent:**', min_value=0.1, max_value=0.9, step=0.1, value=0.6) 
    
    # Draft initial response
    if st.button("Generate Response"):
        with st.spinner("Generating response..."):
            # Select relevant responses
            relevant_responses = select_relevant_responses(demands, examples, email_content, donor_info, actions, additional_messages, additional_guidelines)
            
            # st.subheader("Relevant Response Parts")
            # st.write(relevant_responses)
            
            initial_draft = draft_initial_response(email_content, actions, additional_messages, additional_guidelines, donor_info, relevant_responses, st.session_state.detected_language, name, organization)
           
            # st.subheader("First draft")
            # st.write(initial_draft)

            refined_response = refine_response(initial_draft, donor_info, st.session_state.detected_language, name, organization, set_temperature)
             
            st.session_state.generated_response = refined_response

    # Display generated response
    if st.session_state.generated_response:
        st.subheader("AI Generated Email Response")
        st.text_area("Final Response", value=st.session_state.generated_response, height=500)
        st.write("*If the response doesn't suit you, rerun the tool. In 20% of use cases, AI can lose track and perform below expectations. Or consider adapting your inputs in the text areas.*")
        
        # Translation option
        if st.button("Translate the generated email"):
            st.session_state.translated_response = translate_email(st.session_state.generated_response, st.session_state.detected_language, st.session_state.target_language)

        # Display translated response
        if st.session_state.translated_response:
            st.subheader(f"Translated Reply ({st.session_state.target_language})")
            st.write(st.session_state.translated_response)
            
      
if __name__ == "__main__":
    main()

