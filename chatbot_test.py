from chatbot import QnAChatBot as chatbot


if __name__ == '__main__':
    docs_dir_path = "./document_set"
    db_dir = './database'
    cb = chatbot(docs_dir_path, db_dir)
    answer = cb.ask("음식물이 묻은 비닐 쓰레기는 어떻게 배출해야 돼?")
    print(answer['content'])
    print(f"Sources:\n{"\n".join(answer['source'])}")