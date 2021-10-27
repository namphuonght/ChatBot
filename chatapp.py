import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

#đọc các file vừa trainning
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents1.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# hàm dùng để tiền xử lý các câu đầu vào do người dùng nhập
def clean_up_sentence(sentence):
    # tách từ câu thành các từ
    sentence_words = nltk.word_tokenize(sentence)
    # đem các từ về từ gốc
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# trả về mảng bag of words: 0 or 1 cho mỗi từ trong bag tồn tại trong câu


# hàm dùng để tiền xử lý các câu trong file intents1.json
def bow(sentence, words, show_details=True):
    # tách pattern thành các từ
    sentence_words = clean_up_sentence(sentence)
    # bag of words - ma trận N từ, ma trận từ vựng
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # gán 1 nếu từ hiện tại ở vị trí từ vựng
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


# hàm dùng để dự đoán các từ người dùng nhập vào thuộc lớp   nào đã có trong file intents1.json
def predict_class(sentence, model):
    # lọc ra các dự đoán dưới ngưỡng
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sắp xếp theo độ mạnh của độ chính xác
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# hàm để lấy các câu trả lời từ bot
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# hàm bot trả lời câu hỏi của người dùng nhập vào
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("ChatBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Tạo cửa sổ trò chuyện
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Liên kết thanh cuộn với cửa sổ Trò chuyện
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Nút tạo để gửi tin nhắn
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Tạo hộp để nhập tin nhắn
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("", send)


# Đặt tất cả các thành phần trên màn hình
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()