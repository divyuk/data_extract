import re
from typing import Optional

file_path = 'Notes\All ML Algo\'s links.txt'
lines_list = []

def get_heading(text:str, regex_exp : Optional[str] = r'\w+')->str:
    matched_string = re.findall(regex_exp,text)
    final_string = " ".join(matched_string)
    return final_string

def append_in_file(final_result,index):
    if final_result != '\n' and final_result != '':
        with open('myfile.md', 'a+') as write_file:
            write_file.write(str(index) + ") ")
            write_file.write(final_result)
            write_file.write("\n") 
            final_result = ''
    
sent_list = []
final_result = ''    
index_head  = 0
index_links = 0
bullet = 0

with open(file_path , 'r') as file:
    for line in file.readlines():
        lines_list.append(line)

for singleline in lines_list:
    if '#' in singleline:
        final_result = get_heading(singleline)
        index_head+=1
        index_links=0
        append_in_file(final_result,index_head)
    elif 'https' in singleline:
        final_result = singleline
        index_links+=1
        append_in_file(final_result,index_links)
    else:
        if singleline != '\n':
            sent_list.append(singleline)

for sentence in sent_list:
    if sentence != '\n':
        bullet+=1
        append_in_file(sentence,bullet)