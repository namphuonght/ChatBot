# load các thư viện cần dùng
# thư viện nltk dùng để tiền xử lý dữ liệu như là: xóa các từ stopwords, tách thành các câu từ trong văn bản,...
import nltk
from nltk.stem import WordNetLemmatizer # dùng để đưa các từ về từ gốc(ví dụ dogs->dog,churches->church,..)
import json # thư viện json dùng để load file json
import pickle # thư viện pickle dùng để đọc,ghi file
import numpy as np # thư viện numpy dùng để thao tác với mảng
from keras.models import Sequential # thư viện keras dùng để tạo training và tạo model
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD # SGD(gradient descent) dùng để đánh giá và tối ưu hóa mô hình
import random


lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents1.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tách thành các từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # thêm các từ vào document
        documents.append((w, intent['tag']))

        # thêm vào classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# đưa các từ về từ gốc, ghi thường các chữ, và xóa bỏ các phần tử trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sắp xếp classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

# lưu list word thành file words.pkl, list classes thành file classes.pkl
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# khởi tạo danh sách training
training = []
# tạo một mảng trống với độ dài bằng độ dài của classes list
output_empty = [0] * len(classes)
# training set, bag of words cho mỗi câu
for doc in documents:
    # khởi tạo bag of words
    bag = []
    # danh sách các tokenized words
    pattern_words = doc[0]
    # lemmatize từng từ - tạo từ cơ bản, cố gắng biểu diễn các từ liên quan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # tạo bag of words mảng với index 1, nếu tìm thấy kết hợp từ trong pattern hiện tại
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # đầu ra là '0' cho mỗi tag và '1' cho tag hiện tại(cho mỗi pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# random các features và biến thành mảng np.array
random.shuffle(training)
training = np.array(training)
# tạo danh sách train và kiếm tra. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# tạo model - 3 lớp. lớp đầu tiên với 128 nơ ron, lớp thứ hai với 64 nơ ron and lớp thứ 3 là lớp đầu ra chứa số lượng nơ ron
# bằng số ý định để dự đoán ý định đầu ra với softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# biên dịch model. dùng gradient descent để tối ưu hóa mô hình
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting và lưu models lại
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
