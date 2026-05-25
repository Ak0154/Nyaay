import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(
    question: str,
    context: str
):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"system",
                "content":
                """
                You are a legal assistant.

                Use ONLY the provided context.

                If information is not present,
                say:
                'I could not find this in the documents.'
                """
            },
            {
                "role":"user",
                "content":
                f"""
                Context:
                {context}

                Question:
                {question}
                """
            }
        ]
    )

    return (
        response
        .choices[0]
        .message.content
    )