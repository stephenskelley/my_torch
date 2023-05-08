import pickle
import os
from llama_index import GPTSimpleVectorIndex
from llama_index import download_loader
from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader
import sys
sys.path.insert(0, '../')
import local_secrets as secrets

os.environ['OPENAI_API_KEY'] = secrets.techstyle_openai_key
os.environ['GITHUB_TOKEN'] = secrets.ssk_github_token

download_loader("GithubRepositoryReader")

docs = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "jerryjliu",
        repo =                   "llama_index",
        filter_directories =     (["llama_index", "docs"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".py"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTSimpleVectorIndex.from_documents(docs)

print(index.query("Explain each LlamaIndex class?"))