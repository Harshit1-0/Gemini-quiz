from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(template="You are taking a quiz of the user where you will be provided with topic {topic}. "
                                "You will provide 4 options, and the user will give an answer. "
                                "After giving the answer, you need to provide some details about it and ask the next question \
                                    and if ans is wrong then return the correct option first in bold and big letter and then provide the explaination",
                       input_variables=['topic'])

chat_history = []  

while True:
    topic = input("Enter any topic you want to give a quiz on, or type 'exit' to quit: ")


    if topic == "exit":
        print("Goodbye!")
        break

    generate_quiz = prompt.invoke(topic)
    
    system_message = SystemMessage(content=generate_quiz.text)
    
    chat_history.append(system_message)
  
       
    quiz_response = llm.invoke(generate_quiz.text)
    print(quiz_response.content)
    chat_history.append(AIMessage(quiz_response.content))



    

    while True:
        ans = input("Write your answer (a, b, c, d) or 'exit' to quit: ").lower()

        if ans == "exit":
            print("Goodbye!")
            break

        chat_history.append(HumanMessage(content=ans))

        res_from_ai = llm.invoke(chat_history)
        print(res_from_ai.content) 
        
        chat_history.append(AIMessage(content=res_from_ai.content))
