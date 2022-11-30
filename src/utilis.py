import re
import os
from typing import Optional
from pathlib import Path
from constant import FILEFORMAT,RESULT_DIR

def get_filename(filepath:str):
    created_file = Path(filepath).stem
    return str(created_file)+f'{FILEFORMAT}'
    
def get_text(text:str, regex_exp : Optional[str] = r'\w+')->str:
    matched_string = re.findall(regex_exp,text)
    final_string = " ".join(matched_string)
    return final_string

def repr_html(url,text):
    return f'<a href="{url}">{text}</a>'

def append_in_file(result_filename:str,final_result:str,index:int):
    dir_path = os.path.dirname(f'{RESULT_DIR}/')
    os.makedirs(dir_path,exist_ok=True)
    path_file = f'{dir_path}\{result_filename}'
    if final_result != '\n' and final_result != '':
        with open(path_file, 'a+', encoding="utf8") as write_file:
            write_file.write(str(index) + ". ")
            write_file.write(final_result)
            write_file.write("\n") 
            final_result = ''

def append_text(result_filename:str, space:str, key : Optional[str] = ''):
    dir_path = os.path.dirname(f'{RESULT_DIR}/{result_filename}')
    os.makedirs(dir_path,exist_ok=True)
    path_file = f'{dir_path}\{result_filename}'
    with open(path_file, 'a+', encoding="utf8") as write_file:
        write_file.write(space)
        if key != '':
            write_file.write(key)