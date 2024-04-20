from graphConstructor import build_knowledge_graph, get_bfs_edge_list
from findGoldenEntity import get_most_similar_entity_ids
from promptLLM import ask_LLM
import ast
from tqdm import tqdm

def get_responses_list(line):
    answers_list = [] 
    start_char = 0
    quotation_open = False
    if len(line) == 0:
        return [""]
    # if line[0] != "\"":
    #     print("Error 1: ", line)
    # if line[len(line) - 1] != "\"":
    #     print("Error 2: ", line)
    
    comma_count = 0
    quotation_count = 0
    for i in range(len(line)):
        if line[i] == "\"":
            quotation_count += 1
            if not quotation_open:
                quotation_open = True
                start_char = i
            else:
                answers_list.append(line[start_char + 1: i])
                quotation_open = False
        
        if line[i] == "," and not quotation_open:
            comma_count += 1
    
    # if (2 * (comma_count + 1)) != quotation_count:
    #     print("Error 3: ", quotation_count, comma_count, line)

    return answers_list




def evaluate_performance(output_file_address, expected_answers):
    total = 0
    correct = 0
    false_positive = 0
    false_negative = 0
    precisions = []
    recalls = []
    output_file = open(output_file_address, 'r')
    Lines = output_file.readlines()
    print("total\ttrue\tfp\tfn")
    iterator = 0
    for line in Lines:
        total += len(expected_answers[iterator])
        line_str = line[line.index("[") + 1:line.index("]")]
        response_list = get_responses_list(line_str)
        print("List:\t\t", response_list)
        print("Expected:\t", expected_answers[iterator])


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

        precisions.append(curr_correct / len(response_list))
        recalls.append(curr_correct / len(expected_answers[iterator]))

        print(str(len(expected_answers[iterator])) + "\t" + str(curr_correct) + "\t" + str(curr_fp) + "\t" + str(curr_fn))
        iterator += 1

    print("Overall stats: \nTotal:{} \nCorrect: {}\nFalse Positive: {}\nFalse Negative: {}".format(total, correct, false_positive, false_negative))
    ov_precision = correct / (correct + false_positive)
    ov_recall = correct / (correct + false_negative)
    overall_f1 = 2 * ov_precision * ov_recall / (ov_recall + ov_precision)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = 2 * avg_precision * avg_recall / (avg_recall + avg_precision)

    print("Overall:\nPrecision: {}\tRecall {}\tF1: {}".format(ov_precision, ov_recall, overall_f1))
    print("Question-wise average:\nPrecision: {}\tRecall {}\tF1: {}".format(avg_precision, avg_recall, avg_f1))
    
        

def run_1hop_tests(start, end):
    graph = build_knowledge_graph(edge_list_file = "dataset/MetaQA/MetaQA-3/kb.txt")
    
    one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/1-hop/ntm/qa_test.txt', 'r')
    Lines = one_hop_QA_file.readlines()
    
    expected_answers = []
    end = min(end, len(Lines))
    # counter = -1
    for i in tqdm(range(start, end)):
        # counter += 1
        # if counter >= start and counter < end:
        #     break
        line = Lines[i]
        question = line[:line.find("\t")] + "?"

        expected_answers_list = (line[line.find("\t") + 1:]).split("|")
        expected_answers.append(expected_answers_list)
        named_entity = question[question.find("[") + 1: question.find("]")]
        edge_desc_list = get_bfs_edge_list(graph, named_entity, depth = 1, expand_ending_nodes = False)

        question = question.replace("[", "").replace("]", "")
        LLM_response = ask_LLM(i, edge_desc_list, question)
        
        one_hop_output_file = open('results/1-hop-output.txt', 'a')
        one_hop_output_file.write(LLM_response + "\n")
        one_hop_output_file.close()
    
    # evaluate_performance("results/1-hop-output.txt", expected_answers)



def run_2hop_tests(start, end):
    graph = build_knowledge_graph(edge_list_file = "dataset/MetaQA/MetaQA-3/kb.txt")
    
    one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/2-hop/ntm/qa_test.txt', 'r')
    Lines = one_hop_QA_file.readlines()
    
    expected_answers = []
    end = min(end, len(Lines))

    for i in tqdm(range(start, end)):
        line = Lines[i]
        question = line[:line.find("\t")] + "?"

        expected_answers_list = (line[line.find("\t") + 1:]).split("|")
        expected_answers.append(expected_answers_list)
        named_entity = question[question.find("[") + 1: question.find("]")]
        edge_desc_list = get_bfs_edge_list(graph, named_entity, depth = 2, expand_ending_nodes = False)

        question = question.replace("[", "").replace("]", "")
        LLM_response = ask_LLM(i, edge_desc_list, question)
        
        one_hop_output_file = open('results/2-hop-output.txt', 'a')
        one_hop_output_file.write(LLM_response + "\n")
        one_hop_output_file.close()
    
    # evaluate_performance("results/1-hop-output.txt", expected_answers)


def check_named_entity():    
    one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/1-hop/ntm/qa_test.txt', 'r')
    Lines = one_hop_QA_file.readlines()
    
    wrong_count = 0
    for i in tqdm(range(len(Lines))):
        line = Lines[i]
        question = line[:line.find("\t")] + "?"
        named_entity = question[question.find("[") + 1: question.find("]")]
        _, labels = get_most_similar_entity_ids(query = question, n = 1)
        
        if labels[0] != named_entity:
            wrong_count += 1
            print("############\nNamed Entity: {}".format(named_entity))
            print("\nMost Similar: {}\n############".format(labels[0]))
    
    print("Wrong count:", wrong_count)



# one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/2-hop/ntm/qa_test.txt', 'r')
# Lines = one_hop_QA_file.readlines()
# expected_answers = []
# for line in Lines:
#     line = line.strip()
#     expected_answers_list = (line[line.find("\t") + 1:]).split("|")
#     expected_answers.append(expected_answers_list)

# evaluate_performance("results/2-hop-output.txt", expected_answers)


# run_1hop_tests(9001, 10000)

# check_named_entity()

run_2hop_tests(6001, 7001)
