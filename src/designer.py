from utilis import get_text, append_in_file, append_text, repr_html, get_filename
from custom_exceptions import InvalidFILEException
class DesignFiles:
    """
    This Design the Files in specified format(default:markdown) by taking the Notes in UTF format.
    """
    
    def __init__(self) -> None:
        self.lines_list = []
        self.sent_list = []   
        self.index_head  = 0
        self.index_links = 0
        self.bullet = 0
        self.tabspace = "    "

    def create_file(self , file_path):
        result_filename = get_filename(file_path)
        with open(file_path , 'r', encoding="utf8") as file:
            try:
                for line in file.readlines():
                    self.lines_list.append(line)
            except Exception:
                raise InvalidFILEException
            
        for singleline in self.lines_list:
            if '#' in singleline:
                final_heading = get_text(singleline)
                self.index_head+=1
                self.index_links=0
                append_in_file(result_filename,final_heading,self.index_head)
            elif 'https' in singleline:
                final_link = singleline
                self.index_links+=1
                reg = r'\/([a-z\.0-9_-]*[\/]?\.?[a-z]*?)$'
                final_text = get_text(final_link , reg)
                dot_index = final_text.find('.')

                if final_text == '':
                    final_text = "LINK"
                if dot_index !=-1:
                    final_text = final_text[0:dot_index]
                elif final_text[-1] =='/':
                    final_text = final_text[0:-1] 
        
                html = repr_html(final_link,final_text)
                append_text(result_filename,self.tabspace)
                append_in_file(result_filename,html,self.index_links)
            else:
                if singleline != '\n':
                    self.sent_list.append(singleline)
                    
        append_text(result_filename,space =" ", key="\nKEY POINTS\n")
        for sentence in self.sent_list:
            sentence = sentence.strip()
            if sentence != '':
                self.bullet+=1
                append_in_file(result_filename,sentence,self.bullet)