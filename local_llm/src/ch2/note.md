```bash
$ uv venv --python 3.12
$ uv pip install ollama==0.4.7
$ uv pip install requests
$ uv run ollama serve &
$ uv run ollama pull llama3.2
$ uv run python ollama_requests.py
or
$ uv run ollama_requests.py
```
