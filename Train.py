#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')

#loading the save numpy arrays in the previous code


# In[10]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam ,RMSprop ,SGD ,Nadam ,Adamax
import random

model=Sequential()

model.add(Conv2D(filters=random.randint(100,200),kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
def arch(option):
    if option == 1:
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
    elif option == 2:
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
    elif option == 3:
        #two convolutional and 2 max pooling layers
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    elif option == 4:
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    
    else:
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Conv2D(filters=random.randint(60,100),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPooling2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
arch(random.randint(1,4))  
model.add(Flatten())
model.add(Dropout(0.5))

def fullyconnected(option):
    if option == 1:
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
    elif option == 2:
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
    elif option == 3:
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
    elif option == 4:
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        
    else:
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))
        model.add(Dense(units=random.randint(30,100),activation=random.choice(('relu','sigmoid'))))

fullyconnected(random.randint(1,5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])


# In[11]:


from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)


# In[12]:


checkpoint = ModelCheckpoint('model_best.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=random.randint(1,50),callbacks=[checkpoint],validation_split=0.2)


# In[6]:





# In[7]:





# In[13]:


print(model.evaluate(test_data,test_target))


# In[14]:


history.history


# In[21]:


print(history.history['accuracy'][-5])


# In[22]:


model_architecture =str(model.layers)
accuracy = str(history.history['accuracy'][-5])


# In[24]:


if history.history['accuracy'][-5] >= .90:
    import smtplib
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()

    s.login("rajadityaranjan@gmail.com", "")


    # message
    message1 = accuracy
    message2 = model_architecture

    # sending the mail 
    s.sendmail("rajadityaranjan@gmail.com", "yoyoprinceking@gmail.com", message1)
    s.sendmail("rajadityaranjan@gmail.com", "yoyoprinceking@gmail.com", message2)

    # terminating the session 
    s.quit()


# In[ ]:




