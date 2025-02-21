

EMAIL_IMAGES_DIR = "~/Desktop/example_emails"


from pinecone import Pinecone

from dotenv import load_dotenv
import os

load_dotenv()

os.getenv('PINECONE_API_KEY')

pc = Pinecone(os.getenv('PINECONE_API_KEY'))

##


