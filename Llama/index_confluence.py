import time
import llama_index
from atlassian import Bitbucket
import os
import sys
sys.path.append('../')
import local_secrets as secrets

start_time = time.time()
stash = Bitbucket('https://git.techstyle.net', token=secrets.stash_token)
os.environ['OPENAI_API_KEY'] = secrets.techstyle_openai_key
project ='DATASICENCE'
repo = stash.get_repo(project, 'brand-analytics')
length_cutoff = 100000
for repo in stash.repo_list(project):
    count = 0
    repo_slug = repo['slug']
    files = stash.get_file_list(project, repo_slug)
    index = llama_index.GPTSimpleVectorIndex([])
    index_file = f'./stash_index/{project}_{repo_slug}.json'
    if os.path.isfile(index_file):
        continue
    for file in files:
        if file[-3:] not in ['.py']:
            continue
        try:
            count = count + 1
            url = f"https://git.techstyle.net/projects/{project}/repos/{repo_slug}/browse/{file}"
            code = str(stash.get_content_of_file(project, repo_slug, file))
            code = code[2:len(code)-1].replace("\\n", '\n')
            print(file, len(code))
            if len(code) > length_cutoff:
                print(f'{repo_slug} {file} size {len(code)}, truncating')
                code = code[0:length_cutoff]
            content = f"Stash Project: {project}\nStash Repository: {repo_slug}\nStash URL: {url}\nStash Code:\n {code}"
            index.insert(llama_index.Document(content))
        except Exception as e:
            print(f'Error {e} on {repo_slug} {file}')
    index.save_to_disk(index_file)
    print(f'Done, {count} files in repo {repo_slug} saved to index in {round(time.time() - start_time, 0)} seconds.')

# projects = stash.project_list()
# for project in projects:
#     print(project['key'])
# repos = stash.repo_list('DataScience')
# for repo in repos:
#     print(repo['slug'])
