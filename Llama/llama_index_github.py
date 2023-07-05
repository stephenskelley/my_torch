import pickle
import os
from llama_index import GPTSimpleVectorIndex
from llama_index import download_loader
# from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader
from llama_index import GPTQdrantIndex
import qdrant_client
import sys
sys.path.insert(0, './')
import local_secrets as secrets

os.environ['OPENAI_API_KEY'] = secrets.techstyle_openai_key
os.environ['GITHUB_TOKEN'] = secrets.ssk_github_token

download_loader("GithubRepositoryReader")

docs = None
if os.path.exists("github_llama_index.pkl"):
    with open("github_llama_index.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "jerryjliu",
        repo =                   "llama_index",
        filter_directories =     (['llama_index', 'docs', 'examples'], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = (['.py', '.md', '.ipynb'], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )
    docs = loader.load_data(branch="main")
    with open("github_llama_index.pkl", "wb") as f:
        pickle.dump(docs, f)

client = qdrant_client.QdrantClient(url='http://localhost:6333')
index = GPTQdrantIndex.from_documents(docs, client=client, collection_name='github_llama_index', disallowd_special=())

#index = GPTSimpleVectorIndex.from_documents(docs)
#print(index.query("Explain each LlamaIndex class?"))