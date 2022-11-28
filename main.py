import re
from typing import Optional

file_path = 'Notes\All ML Algos links.txt'
lines_list = []

def get_text(text:str, regex_exp : Optional[str] = r'\w+')->str:
    matched_string = re.findall(regex_exp,text)
    final_string = " ".join(matched_string)
    return final_string

def repr_html(url,text):
    return f'<a href="{url}">{text}</a>'

def append_in_file(final_result,index):
    if final_result != '\n' and final_result != '':
        with open('myfile.md', 'a+') as write_file:
            write_file.write(str(index) + ") ")
            write_file.write(final_result)
            write_file.write("\n") 
            final_result = ''

def append_space(space):
    with open('myfile.md', 'a+') as write_file:
        write_file.write(space)
    
sent_list = []
final_result = ''    
index_head  = 0
index_links = 0
bullet = 0
tabspace = "    "

with open(file_path , 'r') as file:
    for line in file.readlines():
        lines_list.append(line)

for singleline in lines_list:
    if '#' in singleline:
        final_heading = get_text(singleline)
        index_head+=1
        index_links=0
        append_in_file(final_heading,index_head)
    elif 'https' in singleline:
        final_link = singleline
        index_links+=1
        reg = r'\/([a-z\.0-9_-]*[\/]?\.?[a-z]*?)$'
        final_text = get_text(final_link , reg)
        if final_text == ' ':
            print(final_link)
        try :
            if final_text == '':
                final_text = "LINK"
            elif final_text != ' '  or final_text[-1] =='/':
                final_text = final_text[0:-1]
            elif final_text != ' '  or final_text.find('.')!=-1:
                dot_index = final_text.find('.')
                final_text = final_text[0:dot_index]
                
            html = repr_html(final_link,final_text)
            append_space(tabspace)
            append_in_file(html,index_links)
        except Exception as e:
            print(final_text,e)
    else:
        if singleline != '\n':
            sent_list.append(singleline)

for sentence in sent_list:
    if sentence != '\n':
        bullet+=1
        append_in_file(sentence,bullet)