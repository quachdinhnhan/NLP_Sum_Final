

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        with open(self.file_path, 'r',encoding='utf-8') as file:
            doc_file = file.read()
        return doc_file

    