from graphConstructor import build_knowledge_graph, get_bfs_edge_list
from promptLLM import ask_LLM
import ast


def evaluate_performance(output_file_address, expected_answers):
    total = 0
    correct = 0
    false_positive = 0
    false_negative = 0
    output_file = open(output_file_address, 'r')
    Lines = output_file.readlines()
    print("total\ttrue\tfp\tfn")
    iterator = 0
    for line in Lines:
        total += len(expected_answers[iterator])
        line = line[line.index("[")+1:line.index("]")].strip()

        # Splitting the content by commas and then stripping whitespace and both types of quotes
        response_list = [line.strip().strip("\'\"") for item in line.split(",")]
        # line = line[line.find("["):]
        # response_list = line.strip('][').split(', ')
        print("List: ", response_list)
        # response_list = ast.literal_eval(line)
        # response_list = [n.strip() for n in response_list]

        curr_correct = 0
        curr_fp = 0
        for item in response_list:
            if item in expected_answers[iterator]:
                curr_correct += 1
            else: 
                curr_fp += 1
        
        curr_fn = (len(expected_answers[iterator]) - curr_correct)
        correct += curr_correct
        false_positive += curr_fp
        false_negative += curr_fn

        print(str(len(expected_answers[iterator])) + "\t" + str(curr_correct) + "\t" + str(curr_fp) + "\t" + str(curr_fn))
        iterator += 1

    print("Overall stats: \nTotal:{} \nCorrect: {}\nFalse Positive: {}\n False Negative: {}".format(total, correct, false_positive, false_negative))
        




def run1HopTests():
    graph = build_knowledge_graph(edge_list_file = "dataset/MetaQA/MetaQA-3/kb.txt")
    
    one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/1-hop/ntm/qa_test.txt', 'r')
    Lines = one_hop_QA_file.readlines()
    
    expected_answers = []
    count = 0
    for line in Lines:
        if count > 20:
            break
        question = line[:line.find("\t")] + "?"
        expected_answers_list = (line[line.find("\t") + 1:]).split("|")
        expected_answers.append(expected_answers_list)
        named_entity = question[question.find("[") + 1: question.find("]")]
        edge_desc_list = get_bfs_edge_list(graph, named_entity, depth = 1, expand_ending_nodes = False)

        LLM_response = ask_LLM(edge_desc_list, question)
        
        with open('results/1-hop-output.txt', 'a') as one_hop_output_file:
            one_hop_output_file.write(LLM_response)
        count += 1
    
    evaluate_performance("results/1-hop-output.txt", expected_answers)



one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/1-hop/ntm/qa_test.txt', 'r')
Lines = one_hop_QA_file.readlines()
expected_answers = []
for line in Lines:
    question = line[:line.find("\t")] + "?"
    expected_answers_list = (line[line.find("\t") + 1:]).split("|")
    expected_answers.append(expected_answers_list)

evaluate_performance("results/1-hop-output.txt", expected_answers)
# run1HopTests()
