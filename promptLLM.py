
from openai import OpenAI
from graphConstructor import traverse_node_neighborhood


def setup_LLM(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": 
"""You are a QA assistant skilled in answering questions about movies, but not from your own knowledge.
In each prompt, you will be provided with some external information and a user query (question) about movies, directors, actors, and so on. 
The information is retrieved from a Knowledge Graph (KG) representing a movie dataset and contains a number of KG edges in the form of (head, relation, tail). 
You will be asked to answer that query ONLY based on the external information you have been provided. Answers are typically named entities, such as the names of movies, actors, directors, and writers. 
However, questions might have multiple answers. In such cases, form your final answer in the shape of an array. If you find "movie1", "movie2", and "movie3" as the answers to a query, your response should be: ["movie1", "movie2", "movie3"].
                    
Your prompt format will be like: 
########
# KNOWLEDGE GRAPH INFORMATION:
A set of knowledge graph edges retrieved to help you answer the query.


# QUERY
The user's question goes here...
########

Make sure that you have retrieved answers solely based on the provided information from the knowledge graph since you might be asked to explain your reasoning. Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
Your output should ONLY containt the list. Even if it contained one answer, it should be a list of one element. 
"""
            },
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content



def ask_LLM(edge_list, question, expected_answer):
    edge_list_str = ""
    for edge in edge_list:
        edge_list_str += (str(edge) + "\n")

    prompt = """########
# KNOWLEDGE GRAPH INFORMATION:
{}
        
# QUERY
{}
########""".format(edge_list_str, question)
    
    print("Prompt:\n", prompt)
    response = setup_LLM(prompt)
    print("LLM Response:", response)
    

def performQA():
    line = "which person directed the movies starred by [John Krasinski]	Nancy Meyers|Sam Mendes|George Clooney|Ken Kwapis|Luke Greenfield"
    question = line[:line.find("\t")] + "?"
    expected_answers_list = (line[line.find("\t") + 1:]).split("|")
    edge_list = traverse_node_neighborhood("John Krasinski", 2)

    print("Question:", question)
    print("Expected Answers", expected_answers_list)

    ask_LLM(edge_list, question, expected_answers_list)


performQA()

### 1-hop
# [Joe Thomas] appeared in which films	The Inbetweeners Movie|The Inbetweeners 2
# which movies have made [Michelle Trachtenberg] star in	Inspector Gadget|Black Christmas|Ice Princess|Harriet the Spy|The Scribbler
# ce qui fait [Helen Mack] star dans	The Son of Kong|Kiss and Make-Up|Divorce
# The films have done [Shahid Kapoor] to act in	Haider|Jab We Met|Chance Pe Dance
# [Brendan Gleeson] appeared in which films	The Raven|Lake Placid|Calvary|The Smurfs 2|Stonehearst Asylum|The Grand Seduction|The General|Six Shooter|Harrison's Flowers|Into the Storm|The Tiger's Tail
# What the films make [Paresh Rawal] appear in	Table No. 21|Cheeni Kum


### 2-hop
# which person directed the movies starred by [John Krasinski]	Nancy Meyers|Sam Mendes|George Clooney|Ken Kwapis|Luke Greenfield
# who are movie co-directors of [Delbert Mann]	Franco Zeffirelli|Cary Fukunaga|Lewis Milestone|Robert Stevenson
# What are the main languages in [David Mandel] films	German