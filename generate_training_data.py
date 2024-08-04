import pandas as pd
import os
import xml.etree.ElementTree as ET

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

train1_data = r".\TrainingDatasets\TREC-2017-LiveQA-Medical-Train-1.xml"
train2_data = r".\TrainingDatasets\TREC-2017-LiveQA-Medical-Train-2.xml"

# Open the xml files and read the data
with open(train1_data, 'r', encoding='utf-8') as file:
    train1_data = file.read()

with open(train2_data, 'r', encoding='utf-8') as file:
    train2_data = file.read()

def parse_xml_training_data(xml_training_data):
    root = ET.fromstring(xml_training_data)

    # Extract data
    questions_data = []
    for question in root.findall('NLM-QUESTION'):
        question_id = question.get('questionid')
        f_ref = question.get('fRef')
        subject = question.find('SUBJECT').text if question.find('SUBJECT') is not None else ""
        question_text = question.find('MESSAGE').text if question.find('MESSAGE') is not None else ""
        
        for sub_question in question.find('SUB-QUESTIONS').findall('SUB-QUESTION'):
            sub_question_id = sub_question.get('subqid')
            focus = sub_question.find('ANNOTATIONS').find('FOCUS').text
            q_type = sub_question.find('ANNOTATIONS').find('TYPE').text
            
            for answer in sub_question.find('ANSWERS').findall('ANSWER'):
                answer_id = answer.get('answerid')
                pair_id = answer.get('pairid')
                answer_text = answer.text
                
                questions_data.append({
                    'question_id': question_id,
                    'f_ref': f_ref,
                    'subject': subject,
                    'question': question_text,
                    'sub_question_id': sub_question_id,
                    'focus': focus,
                    'type': q_type,
                    'answer_id': answer_id,
                    'pair_id': pair_id,
                    'answer_text': answer_text
                })

    return pd.DataFrame(questions_data)

def main():
    df_train1 = parse_xml_training_data(train1_data)
    df_train2 = parse_xml_training_data(train2_data)

    # Join the two training datasets
    df_train = pd.concat([df_train1, df_train2], ignore_index=True)

    # Save the data to csv
    df_train1.to_csv(r".\TrainingDatasets\TREC-2017-LiveQA-Medical-Train-1.csv", index=False)
    df_train2.to_csv(r".\TrainingDatasets\TREC-2017-LiveQA-Medical-Train-2.csv", index=False)
    df_train.to_csv(r".\TrainingDatasets\TREC-2017-LiveQA-Medical-Train.csv", index=False)
    print(df_train1.shape)
    print(df_train2.shape)
    print(df_train.shape)

if __name__ == "__main__":
    main()
