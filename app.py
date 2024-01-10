import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI


default_url = 'https://www.youtube.com/watch?v=xZIB8vauWSI&ab_channel=CarolMeierNarrator-revoeciov'


prompt_template = ''' 
Write a summary of the following:
"{text}"

Additionally, highlight any noteworthy and interesting aspects from the text. The summary should cover the key points, while also including any interesting details. Remember to focus on clarity, brevity and interesting points when providing the summary.

Also add '\n' in your answer as output will be used in python prrint function:

CONCISE SUMMARY:'''
prompt = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm_chain = LLMChain(llm=llm, prompt=prompt)

progress_bar = st.empty()
def main():
    st.title('🎥 YouTube Video Summary Generator 📝')
    openai_key = st.text_input('🔑 Enter your OpenAI Key:', type='password')
    input_url = st.text_input('🔗 Enter YouTube video link:',default_url)
    summary_button = st.button('🚀 Get Summary')
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        loader = YoutubeLoader.from_youtube_url(input_url, add_video_info=False)
        
        if summary_button:
            progress_bar.progress(0)

            docs = loader.load()

            if not docs:
                st.error("⚠️ Failed to load the trascipt. Please try a different video.")
                progress_bar.empty()
                return
            
            progress_bar.progress(50)

            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            output_text = stuff_chain(docs)['output_text']
            progress_bar.progress(100)
            st.success('✨ Summary Ready!')

            st.write(output_text)
            progress_bar.empty()

if __name__ == '__main__':
    main()