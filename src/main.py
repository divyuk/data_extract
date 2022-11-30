from designer import DesignFiles
from constant import NOTES_DIR
import os
from pathlib import Path

file_list = os.listdir(NOTES_DIR)
for single_file in file_list:
    myfile = DesignFiles()
    file_path=os.path.join(NOTES_DIR,single_file)
    path = Path(file_path)
    myfile.create_file(file_path=path)