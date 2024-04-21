from graphConstructor import build_knowledge_graph, get_bfs_edge_list
from findGoldenEntity import get_most_similar_entity_ids
from promptLLM import ask_LLM
import ast
from tqdm import tqdm
import pandas as pd

def get_responses_list(line):
    answers_list = [] 
    start_char = 0
    quotation_open = False
    if len(line) == 0:
        return [""]
    if line[0] != "\"":
        print("Error 1: ", line)
    if line[len(line) - 1] != "\"":
        print("Error 2: ", line)
    
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
    
    if (2 * (comma_count + 1)) != quotation_count:
        print("Error 3: ", quotation_count, comma_count, line)

    return answers_list


def get_expected_answers_list(file_path):
    file = open(file_path, 'r')
    Lines = file.readlines()
    expected_answers = []

    for line in Lines:
        line_str = line.strip()
        expected_answers.append(line_str.split("|"))
    
    return expected_answers



def evaluate_performance(output_file_path, expected_answers_path):
    total = 0
    correct = 0
    false_positive = 0
    false_negative = 0
    precisions = []
    recalls = []
    expected_answers = get_expected_answers_list(expected_answers_path)
    cw_error_count = 0
    output_file = open(output_file_path, 'r')
    Lines = output_file.readlines()
    print("total\ttrue\tfp\tfn")
    iterator = 0
    for line in Lines:
        if (line.strip() == "Context Window Error"):
            cw_error_count += 1
            iterator += 1
            continue
        total += len(expected_answers[iterator])
        line_str = line[line.index("[") + 1:line.index("]")]
        response_list = get_responses_list(line_str)
        response_set = set(response_list)
        print("List:\t\t", response_set)
        print("Expected:\t", expected_answers[iterator])


        curr_correct = 0
        curr_fp = 0
        for item in response_set:
            if item in expected_answers[iterator]:
                curr_correct += 1
            else: 
                curr_fp += 1
        
        curr_fn = (len(expected_answers[iterator]) - curr_correct)
        correct += curr_correct
        false_positive += curr_fp
        false_negative += curr_fn

        precisions.append(curr_correct / len(response_set))
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
    print("Context Window Errors: ", cw_error_count)
    



def get_questions_df(file_path):
    columns = ["question", "named_entity", "expected_answers"]

    three_hop_QA_file = open(file_path, 'r')
    Lines = three_hop_QA_file.readlines()

    df = pd.DataFrame(columns=columns)
    
    for line in Lines:
        line_str = line.strip()
        question = line_str[:line_str.find("\t")] + "?"

        expected_answers = line_str[line_str.find("\t") + 1:]
        named_entity = question[question.find("[") + 1: question.find("]")]

        question = question.replace("[", "").replace("]", "")
        df.loc[len(df.index)] = [question, named_entity, expected_answers]

    return df


def run_sampled_tests(num_hops, sample_size = 1000):
    question_file_path = "dataset/MetaQA/MetaQA-3/" + str(num_hops) + "-hop/vanilla/qa_test.txt"
    expected_answers_file_path = "results/" + str(num_hops) + "-hop-expected.txt"
    output_file_path = "results/" + str(num_hops) + "-hop-sample-output.txt"
    log_file_path = "results/" + str(num_hops) + "-hop-sample-log.txt"
    
    graph = build_knowledge_graph(edge_list_file = "dataset/MetaQA/MetaQA-3/kb.txt")
    
    questions_df = get_questions_df(question_file_path)
    sample_df = questions_df.sample(n = sample_size, random_state=1)

    expected_answers = []

    for i in tqdm(range(sample_size)):
        
        question = sample_df.iloc[i]["question"]
        named_entity = sample_df.iloc[i]["named_entity"]
        expected_answers = sample_df.iloc[i]["expected_answers"]

        edge_desc_list = get_bfs_edge_list(graph, named_entity, depth = num_hops, expand_ending_nodes = False)

        try:
            LLM_response = ask_LLM(i, edge_desc_list, question, log_file_path)
            output_file = open(output_file_path, 'a')
            output_file.write(LLM_response + "\n")
            output_file.close()

            expected_file = open(expected_answers_file_path, 'a')
            expected_file.write(expected_answers + "\n")
            expected_file.close()
    
        except:
            output_file = open(output_file_path, 'a')
            output_file.write("Context Window Error\n")
            output_file.close()

            expected_file = open(expected_answers_file_path, 'a')
            expected_file.write(expected_answers + "\n")
            expected_file.close()
    
    # evaluate_performance("results/3-hop-output.txt", 'results/3-hop-expected.txt')


def check_named_entity():    
    one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/1-hop/vanilla/qa_test.txt', 'r')
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



# one_hop_QA_file = open('dataset/MetaQA/MetaQA-3/2-hop/vanilla/qa_test.txt', 'r')
# Lines = one_hop_QA_file.readlines()
# expected_answers = []
# for line in Lines:
#     line = line.strip()
#     expected_answers_list = (line[line.find("\t") + 1:]).split("|")
#     expected_answers.append(expected_answers_list)

# evaluate_performance("results/2-hop-output.txt", expected_answers)

# check_named_entity()


# run_sampled_tests(3, 400)

evaluate_performance("results/3-hop-sample-output.txt", "results/3-hop-expected.txt")
