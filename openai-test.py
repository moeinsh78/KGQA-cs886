from openai import OpenAI



client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a QA assistant, skilled in answering questions about movies and actors. Just give short single answers to the questions."},
        {"role": "user", "content": "what were the release dates of [Robert Adetuyi] directed films?"}
    ]
)

print(completion.choices[0].message.content)