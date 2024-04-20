
from openai import OpenAI
from graphConstructor import *


def setup_LLM(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": 
"""You are a QA assistant skilled in answering questions about movies.
In each input, you will be provided with some external information (INFORMATION) and a user query (QUESTION) about movies. 
The information is retrieved from a Knowledge Graph (KG) representing a movie dataset, but you don't need all of its information to answer the query. Just look for the information that helps you find the answer to the QUESTION.
You will be asked to answer the query ONLY based on the external information you have been provided. The QUESTION might have multiple answers. In such cases, form your final answer in the shape of an array with no additional character in between. Don't include any quotation marks, and output all possible answers in a single array in one line. 

For example, if you find Movie1, Movie2, and Movie3 as the answers to a query, your response should be: 
["Movie1", "Movie2", "Movie3"] 
                    
The input will be bounded by two Hashtag blocks:
######
# INFORMATION:
INFORMATION

# QUERY
QUESTION
######


Your response should ONLY contain the list. Even if there was a single answer, it should be a list of one element. 
Make sure that you have retrieved answers solely based on the provided information from the knowledge graph since you might be asked to explain your reasoning. 
Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
Remember to return just a list of answers and no extra character.
"""
            },
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

# In such cases, form your final answer in the shape of an array. For example, if you find "movie1", "movie2", and "movie3" as the answers to a query, your response should be: 
# ["movie1", "movie2", "movie3"]
# with no additional character in between.
# # For example, let's say you have been asked this question:
# which are the directors of the films written by the writer of [The Green Mile]?

# For this question, you first have to (1) find the writer of the movie The Green Mile. 
# (2) you should look for other movies that this writer has written. 
# (3) you should output the director of those films. 

# Always provide the answers for parts (1), (2), and (3) in the output.
# Explain your thought process in the output: 
# # Part (1) thoughts

# # Part (2) thoughts

# # Part (3) thoughts

# # Final result as a list


def ask_LLM(attempt_number, edge_desc_list, question):
    edge_description_str = ""
    for edge_desc in edge_desc_list:
        edge_description_str += (edge_desc + "\n")

    prompt = """######
# INFORMATION:
{}

# QUERY
{}
######""".format(edge_description_str, question)
    
    response = setup_LLM(prompt)
    log_file = open('results/2-hop-log.txt', 'a')
    log_file.write("######## Question Number: {} ########\n".format(attempt_number + 1))
    log_file.write("Prompt:\n{}\n".format(prompt))
    log_file.write("Response:\n{}\n".format(response))
    log_file.write("#####################################\n")
    log_file.close()
    return response
    

# def perform_QA(edge_description_list, question, expected_answers_list):


#     print("Question:", question)
#     print("Expected Answers", expected_answers_list)

#     return ask_LLM(edge_description_list, question, expected_answers_list)



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


### 3-hop
# the films that share the directors with the film [Catch Me If You Can] were in which languages	German|Polish|Mende|Japanese
# which holds films for the director of [Written on the Wind]	Sandra Dee|Charles Coburn|Cornel Wilde|John Gavin|Warren William|Susan Kohner|Joan Bennett|Fred MacMurray|Barbara Stanwyck|Don Ameche|Jane Wyman|Rochelle Hudson|Boris Karloff|Patricia Knight|Robert Cummings|Rock Hudson|Lucille Ball|Claudette Colbert|Lana Turner|George Sanders
# the films that share actors with the film [Creepshow] were in which languages	Polish|English
# what was the years of release of films that share writers with the film [Grown Ups 2]	1995|1996|1999|1998|1989|2002|2000|2008|2011|2010