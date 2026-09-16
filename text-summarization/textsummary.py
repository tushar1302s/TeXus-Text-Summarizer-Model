import torch
import gradio as gr

#Use a pipeline for text summarization
from transformers import pipeline

model_path = "../Models/distilbart-cnn-12-6"

text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.float16)

# text = """The Tata Group is one of India's oldest and most diversified business groups, with its origins going back to 1868 when Jamsetji Tata established a trading firm. Over the years, the group expanded from its early industrial and trading activities into a large global enterprise operating across many industries. Today, the Tata Group comprises 31 companies and operates in more than 100 countries across six continents. Its businesses cover areas such as information technology, automotive, steel, consumer products and retail, infrastructure, financial services, aerospace and defence, tourism and travel, telecommunications and media, chemicals, and trading and investments. Tata Sons serves as the principal investment holding company and promoter of Tata companies. Approximately 66 percent of the equity share capital of Tata Sons is held by philanthropic trusts associated with the Tata Group, which support activities in areas such as education, healthcare, livelihoods, arts and culture.

# The history of the Tata Group is closely connected with the development of modern Indian industry. Jamsetji Tata had a vision of building industries and institutions that could contribute to India's economic and intellectual development. The group established several important enterprises during the early twentieth century. Tata Steel was established in 1907 and became an important part of India's industrial development. Tata Power developed from Jamsetji Tata's vision of generating clean energy for Mumbai. The group also entered the hospitality sector with the opening of the Taj Mahal Palace Hotel in Mumbai in 1903. Over the following decades, Tata companies expanded into engineering, automobiles, consumer products, chemicals, beverages, telecommunications, financial services and other sectors.

# Tata Consultancy Services, commonly known as TCS, was established in 1968 and became an important part of India's information technology industry. TCS grew from a technology services company into a global organisation providing consulting, technology and business solutions to organisations around the world. Tata Motors developed from the group's engineering and locomotive activities and later became a major automobile manufacturer. The company expanded from commercial vehicles into passenger vehicles and eventually developed a global automotive presence. Tata Steel also expanded internationally and became an important steel producer with operations in several countries.

# The Tata Group has also developed a significant presence in the consumer and retail sector. Companies associated with the group operate brands and businesses involving jewellery, watches, food and beverages, consumer products and retail. Titan has developed businesses in watches, jewellery and other lifestyle products, while Tata Consumer Products operates in food and beverage categories. The group also operates large retail businesses through companies such as Trent. These businesses serve millions of consumers and have become an important part of the group's presence in the Indian consumer market.

# Another major area of the Tata Group is aviation and travel. The group has a long association with Indian aviation, and Tata companies have been involved in the development of airline businesses. The group also operates hotels and hospitality businesses through Indian Hotels Company Limited, whose Taj brand is internationally recognised. In addition, Tata Communications operates in telecommunications and digital infrastructure, connecting businesses and customers across different markets.

# The Tata Group has continued to invest in newer industries and technologies. In recent years, Tata companies have expanded into areas such as electronics manufacturing, semiconductor-related activities, battery manufacturing, telecommunications infrastructure, digital businesses and modern retail. These investments reflect the group's efforts to participate in industries that are becoming increasingly important to the global economy. The group also has businesses involved in aerospace and defence, where technology, engineering and advanced manufacturing play an important role.

# An important characteristic of the Tata Group is its decentralised structure. Tata companies operate independently under their respective boards of directors, while Tata Sons acts as the principal investment holding company and promoter. This structure allows individual companies to focus on their specific industries while remaining connected to the broader Tata Group ecosystem. The group has also maintained a strong association with philanthropy through the Tata Trusts. The trusts support initiatives involving education, healthcare, social development, livelihoods, culture and other areas.

# In financial terms, the Tata Group is a major global business organisation. Tata states that the aggregate revenue of its companies exceeded 180 billion US dollars in the financial year 2024-25, and the companies collectively employ more than one million people. The group had 26 publicly listed Tata companies with an aggregate market capitalisation of more than 328 billion US dollars as of March 31, 2025. These figures demonstrate the scale and diversity of the Tata ecosystem.

# The Tata Group's long history shows how a business organisation can evolve across generations while entering new industries and markets. From its beginnings as a trading enterprise in the nineteenth century, the group developed businesses in steel, energy, automobiles, hospitality, information technology, consumer products, telecommunications, financial services, aviation and advanced manufacturing. Its companies continue to operate in India and international markets, making the Tata Group an important part of India's corporate and industrial landscape."""

# print(text_summary(text,
#     max_length=150,
#     min_length=50))

def summary(input_text):
    output = text_summary(input_text, max_length=100, min_length=50)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summary, inputs=gr.Textbox(lines=10, placeholder="Enter text to summarize..."), outputs=gr.Textbox(lines=5, placeholder="Summarized text generation..."), title="TeXum : Text Summarizer", description="This is a text summarization model that takes a long piece of text as input and generates a concise summary of the text.")
demo.launch(share=True)