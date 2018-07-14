

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

    Using TensorFlow backend.
    


```python
from keras.models import load_model
model = load_model('ideal/ideal.h5')

with open('ideal/model_label_map.dat','rb') as f:
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
total_letters = 6000
total_errors = sum(errors_matrix)
total_errors
```




    201




```python
perc_error = total_errors/total_letters*100
perc_error
```




    3.35




```python
perc_accuracy = 100-perc_error
perc_accuracy
```




    96.65




```python
hypothetical_success_by_image = (perc_accuracy/100)**6*100
hypothetical_success_by_image
```




    81.51004824408923



# Selected Examples


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
      <td>gxozcc</td>
      <td>gxozcc</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nphxvn</td>
      <td>nphxvn</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>keezhu</td>
      <td>koezhu</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>euhybv</td>
      <td>euhybv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>akpvgo</td>
      <td>akpvgo</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ynuazu</td>
      <td>youszu</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sdvxbn</td>
      <td>sdvxbn</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>kwyyge</td>
      <td>kwyyge</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>besgnz</td>
      <td>besgnz</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nbpxzv</td>
      <td>nbpxzv</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


