

```python
import os.path
from imutils import paths, resize
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

from custom import get_letters_list2, normalize_image_size

import pickle

import pandas as pd
from random import choices
import matplotlib.pyplot as plt
```


```python
data = []
labels = []
count = 0
for letter_path in paths.list_images('grainy_samples/letters'):
    image = cv2.imread(letter_path,0)
    
    # Resize image
    image = normalize_image_size(image,20,20)

    
    image = np.expand_dims(image, axis = 2)
    data.append(image)
    labels.append(letter_path.split(os.path.sep)[-2])
    count+=1
    if count%2000==0:
        print('Added image #%d'%count)

```

    Added image #2000
    Added image #4000
    Added image #6000
    Added image #8000
    Added image #10000
    Added image #12000
    Added image #14000
    Added image #16000
    Added image #18000
    Added image #20000
    Added image #22000
    Added image #24000
    Added image #26000
    Added image #28000
    Added image #30000
    Added image #32000
    Added image #34000
    Added image #36000
    Added image #38000
    Added image #40000
    Added image #42000
    Added image #44000
    Added image #46000
    Added image #48000
    Added image #50000
    Added image #52000
    Added image #54000
    Added image #56000
    Added image #58000
    Added image #60000
    Added image #62000
    Added image #64000
    Added image #66000
    Added image #68000
    Added image #70000
    Added image #72000
    Added image #74000
    Added image #76000
    Added image #78000
    Added image #80000
    Added image #82000
    Added image #84000
    Added image #86000
    Added image #88000
    Added image #90000
    Added image #92000
    Added image #94000
    


```python
data = np.array(data,dtype='float')/255.0
labels = np.array(labels)
```


```python
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.25)
```


```python
lb = LabelBinarizer().fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)
```


```python
model = Sequential()
model.add(Conv2D(10, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(40, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(19, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=19, epochs=2, verbose=1)
```

    Train on 71238 samples, validate on 23746 samples
    Epoch 1/2
    71238/71238 [==============================] - 102s 1ms/step - loss: 0.1771 - acc: 0.9529 - val_loss: 0.0593 - val_acc: 0.9846
    Epoch 2/2
    71238/71238 [==============================] - 115s 2ms/step - loss: 0.0481 - acc: 0.9870 - val_loss: 0.0553 - val_acc: 0.9864
    




    <keras.callbacks.History at 0x1bda8327be0>




```python
model.save('models/ideal/ideal.h5')

with open('models/ideal/model_label_map.dat','wb') as f:
    pickle.dump(lb,f)
```


```python
from keras.models import load_model
model = load_model('models/ideal/ideal.h5')

with open('models/ideal/model_label_map.dat','rb') as f:
    lb = pickle.load(f)
```


```python
test_list_full = os.listdir('grainy_samples/samples')
errors_matrix = []


for i in range(100):
    test_list = choices(test_list_full,k=10)
    passwords_list = []
    for test_path in test_list:
        letters_list = get_letters_list2('grainy_samples/samples/%s'%test_path)
        if(type(letters_list[0])==str):
            break
        attempted_password = ''
        for letter_image in letters_list:
            letter_image = normalize_image_size(letter_image,20,20)
    #         cv2.imshow('letter',letter_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
            letter_image = np.expand_dims(letter_image,axis=2)
            letter_image = np.expand_dims(letter_image,axis=0)
            prediction = model.predict(letter_image)
            attempted_password += lb.inverse_transform(prediction)[0]
        passwords_list.append(attempted_password)
    filtered_answers = [a.split('.')[0] for a in test_list]
#     print(len(passwords_list))
    if(len(passwords_list)==len(filtered_answers)):
        for k in range(10):
            errors = sum([passwords_list[k][a]!=filtered_answers[k][a] for a in range(6)])
            errors_matrix.append(errors)
    print('Completed run #%d'%(i+1))
```

    Completed run #1
    Completed run #2
    Completed run #3
    Completed run #4
    Completed run #5
    Completed run #6
    Completed run #7
    Completed run #8
    Completed run #9
    Completed run #10
    Completed run #11
    Completed run #12
    Completed run #13
    Completed run #14
    Completed run #15
    Completed run #16
    Completed run #17
    Completed run #18
    Completed run #19
    Completed run #20
    Completed run #21
    Completed run #22
    Completed run #23
    Completed run #24
    Completed run #25
    Completed run #26
    Completed run #27
    Completed run #28
    Completed run #29
    Completed run #30
    Completed run #31
    Completed run #32
    Completed run #33
    Completed run #34
    Completed run #35
    Completed run #36
    Completed run #37
    Completed run #38
    Completed run #39
    Completed run #40
    Completed run #41
    Completed run #42
    Completed run #43
    Completed run #44
    Completed run #45
    Completed run #46
    Completed run #47
    Completed run #48
    Completed run #49
    Completed run #50
    Completed run #51
    Completed run #52
    Completed run #53
    Completed run #54
    Completed run #55
    Completed run #56
    Completed run #57
    Completed run #58
    Completed run #59
    Completed run #60
    Completed run #61
    Completed run #62
    Completed run #63
    Completed run #64
    Completed run #65
    Completed run #66
    Completed run #67
    Completed run #68
    Completed run #69
    Completed run #70
    Completed run #71
    Completed run #72
    Completed run #73
    Completed run #74
    Completed run #75
    Completed run #76
    Completed run #77
    Completed run #78
    Completed run #79
    Completed run #80
    Completed run #81
    Completed run #82
    Completed run #83
    Completed run #84
    Completed run #85
    Completed run #86
    Completed run #87
    Completed run #88
    Completed run #89
    Completed run #90
    Completed run #91
    Completed run #92
    Completed run #93
    Completed run #94
    Completed run #95
    Completed run #96
    Completed run #97
    Completed run #98
    Completed run #99
    Completed run #100
    


```python
total_errors = sum(errors_matrix)
total_errors
```




    191




```python
total_letters = 6000
```


```python
perc_error = total_errors/total_letters*100
perc_error
```




    3.183333333333333




```python
perc_accuracy = 100-perc_error
perc_accuracy
```




    96.81666666666666




```python
hypothetical_success_by_image = (perc_accuracy/100)**6*100
hypothetical_success_by_image
```




    82.35704518396281



# Selected Examples:


```python
errors_matrix = []

test_list = choices(test_list_full,k=10)
passwords_list = []
for test_path in test_list:
    letters_list = get_letters_list2('grainy_samples/samples/%s'%test_path)
    if(type(letters_list[0])==str):
        break
    attempted_password = ''
    for letter_image in letters_list:
        letter_image = normalize_image_size(letter_image,20,20)
#         cv2.imshow('letter',letter_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        letter_image = np.expand_dims(letter_image,axis=2)
        letter_image = np.expand_dims(letter_image,axis=0)
        prediction = model.predict(letter_image)
        attempted_password += lb.inverse_transform(prediction)[0]
    passwords_list.append(attempted_password)
filtered_answers = [a.split('.')[0] for a in test_list]
#     print(len(passwords_list))
if(len(passwords_list)==len(filtered_answers)):
    for k in range(10):
        errors = sum([passwords_list[k][a]!=filtered_answers[k][a] for a in range(6)])
        errors_matrix.append(errors)
```


```python
df = pd.DataFrame({
    'Attempted Password:': passwords_list,
    'Answers:': filtered_answers,
    'Number of errors': errors_matrix
})
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Answers:</th>
      <th>Attempted Password:</th>
      <th>Number of errors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>zwudux</td>
      <td>zwudux</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hhzxnw</td>
      <td>hhzxnw</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>qxhece</td>
      <td>qxhece</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cqeddw</td>
      <td>cqeddw</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>geacgo</td>
      <td>geocgo</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>kcnncp</td>
      <td>kcnncp</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ngpncq</td>
      <td>ngpncq</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ogopaz</td>
      <td>ogopaz</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ycoxup</td>
      <td>ycoxup</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gqdpdk</td>
      <td>gqdpdk</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


