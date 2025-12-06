from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(template="You are taking a quiz of the user where you will be provided with topic {topic}. "
                                "You will ask the question and you will provide 4 options, and the user will give an answer. "
                                "After giving the answer, you need to provide some details about it and ask the next question. "
                                "If the answer is wrong, return the correct option first in **bold** and **uppercase**, then provide the explanation.",
                       input_variables=['topic'])

chat_history = []

def validate_answer(answer):
    if answer not in ['a', 'b', 'c', 'd']:
        print("Invalid answer! Please choose one of the options (a, b, c, or d).")
        return False
    return True

while True:
    topic = input("Enter any topic you want to give a quiz on, or type 'exit' to quit: ")

    if topic.lower() == "exit":
        print("Goodbye!")
        break

    generate_quiz = prompt.invoke(topic)
    system_message = SystemMessage(content=generate_quiz.text)
    chat_history.append(system_message)

    quiz_response = llm.invoke(generate_quiz.text)
    print(quiz_response.content)
    chat_history.append(AIMessage(content=quiz_response.content))

    while True:
        ans = input("Write your answer (a, b, c, d) or 'exit' to quit: ").lower()

        if ans == "exit":
            print("Goodbye!")
            break

        if not validate_answer(ans):
            continue

        chat_history.append(HumanMessage(content=ans))

        res_from_ai = llm.invoke(chat_history)
        print(res_from_ai.content)
        
        chat_history.append(AIMessage(content=res_from_ai.content))

        next_question = input("Would you like to continue to the next question? (yes/no): ").lower()
        if next_question == "no":
            print("Goodbye!")
            break
