import os
from langchain.text_splitter import CharacterTextSplitter

#define the directory containing text file and persisting directory

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "tbateVOL8.pdf")
persistant_directoryy = os.path.join(current_dir, "db", "chroma_db")
