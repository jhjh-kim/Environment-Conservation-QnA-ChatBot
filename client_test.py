import requests

while True:
    question = input()
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question":question})

    answer = response.json()['content']
    source = response.json()['source']
    print(f"{answer}\nSources:\n{"\n".join(source)}")
