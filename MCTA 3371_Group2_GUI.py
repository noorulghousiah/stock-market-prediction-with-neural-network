# import modules
import torch
import torch.nn as nn
from tkinter import *
from tkscrolledframe import ScrolledFrame

######
class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 4
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 4 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1) 
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
       
    def predict(self,xPredicted, xPredicted_max):
        weekvalue=xPredicted*xPredicted_max
        Label(ws, text=f'Monday to Thursday Stock Prices  : {weekvalue}', pady=15, bg='#ffc9e1').pack()
        ss=(self.forward(xPredicted)* y_max)
        Label(ws, text=f'Predicted Stock Price on Friday : {ss}', pady=15, bg='#ffc9e1').pack()


######

e10=426.92
#d10=426.53
#c10=429.97
#b10=410.22
#a10=394.82

e9=387.70
d9=385.10
c9=374.75
b9=386.50
a9=391.67

e8=389.42
d8=379.76
c8=305.35
b8=306.85
a8=311.73

e7=312.61
d7=316.75
c7=301.75
b7=292.10
a7=289.50

e6=283.37
d6=285.75
c6=288.82
b6=285.68
a6=291.48

e5=286.77
d5=275.59
c5=277.99
b5=282.07
a5=289.07

e4=277.46
d4=272.23
c4=269.53
b4=262.38
a4=270.39

e3=271.16
d3=271.01
c3=269.99
b3=276.64
a3=269.99

e2=267.55
d2=264.60
c2=264.92
b2=271.66
a2=275.76

e1=277.74 #friday
d1=273.80 #thursday
c1=269.81 #wednesday
b1=264.07 #tuesday
a1=265.28 #monday

#data
X = torch.tensor(([a1,b1,c1,d1], [a2,b2,c2,d2], [a3,b3,c3,d3],[a4,b4,c4,d4], [a5,b5,c5,d5], [a6,b6,c6,d6], [a7,b7,c7,d7], [a8,b8,c8,d8], [a9,b9,c9,d9]), dtype=torch.float) # 9 X 4 tensor
y = torch.tensor(([e1], [e2], [e3],[e4], [e5], [e6], [e7], [e8], [e9]), dtype=torch.float) # 10 X 1 tensor


#####
# scale units , need to scale/normalise our data so the range same as transfer function, so can be compared
X_max, _ = torch.max(X, 0) #0 take max in row, 1 take max in column

X = torch.div(X, X_max) #this code will scale data to be in range of 0-1

y_max, _ = torch.max(y, 0) #0 take max in row, 1 take max in column
y = torch.div(y, y_max) #this code will scale data to be in range of 0-1

#####

#training
NN = Neural_Network()
for i in range(10000):  # trains the NN 10,000 times
##    if (i % 100) == 0:
##        print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)


#--------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------GUI-----------------------------------------------------------------------------------------------------------------------------#

# configure workspace
#####
def on_configure(event):
    # update scrollregion after starting 'mainloop'
    # when all widgets are in canvas
    canvas.configure(scrollregion=canvas.bbox('all'))

root = Tk()

#create canvas with scrollbar
canvas = Canvas(root, bg='#fffad2')
canvas.pack(side=LEFT, fill = BOTH, expand = True)

scrollbar = Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=LEFT, fill='y')

canvas.configure(yscrollcommand = scrollbar.set)

# update scrollregion after starting 'mainloop'
# when all widgets are in canvas
canvas.bind('<Configure>', on_configure)

#put frame in canvas
ws = Frame(canvas, bg='#fffad2')
canvas.create_window((0,0), window=ws, anchor='nw')

#####


# function territory
def inputs():
    outputlb=Label(ws, text="The predicted NVIDIA stock price on Friday ", bg='#c6d76b').pack()
    m0 = float(monday.get())
    t0=float(tuesday.get())
    w0=float(wednesday.get())
    k0=float(thursday.get())
    
    xPredicted = torch.tensor(([m0,t0,w0,k0]), dtype=torch.float) # 1 X 4 tensor
    xPredicted_max, _ = torch.max(xPredicted, 0)
    xPredicted = torch.div(xPredicted, xPredicted_max)
    NN.predict(xPredicted, xPredicted_max)
    empty6=Label(ws, text="  ", pady=7, padx=10,  bg='#fffad2').pack()


# label & Entry boxes territory
topic1 =Label(ws, text=" _Stock Prices Prediction on NVIDIA Corporation using Neural Network on Feed Forward Method_", pady=15,padx=30, bg='#fffad2')
topic2 =Label(ws, text=" _Stock Price on Friday_", pady=15, bg='#fffad2')
subtopic= Label(ws, text="Enter stock price of NVIDIA from Monday until Thursday to predict the stock price on Friday", pady=15, bg='#fffad2')

empty1=Label(ws, text="  ", pady=7, padx=10,  bg='#fffad2')
empty2=Label(ws, text="  ", pady=7, padx=10,  bg='#fffad2')
empty3=Label(ws, text="  ", pady=7, padx=10,  bg='#fffad2')
empty4=Label(ws, text="  ", pady=7, padx=10,  bg='#fffad2')
empty5=Label(ws, text="  ", pady=7, padx=10,  bg='#fffad2')


lbmonday = Label(ws, text="Stock Price on Monday?",  padx=10, bg='#fffad2')
monday = Entry(ws)

lbtuesday = Label(ws, text="Stock Price on Tuesday?",  padx=10, bg='#fffad2')
tuesday = Entry(ws)

lbwednesday = Label(ws, text="Stock Price on Wednesday? ", padx=10, bg='#fffad2')
wednesday = Entry(ws)

lbthursday = Label(ws, text="Stock Price on Thursday?", padx=10, bg='#fffad2')
thursday = Entry(ws)


# button territory
welBtn = Button(ws, text="Predict!", command=inputs)


# Position Provide territory
topic1.pack()
topic2.pack()
subtopic.pack()

lbmonday.pack()
monday.pack()
empty1.pack()
lbtuesday.pack()
tuesday.pack()
empty2.pack()
lbwednesday.pack()
wednesday.pack()
empty3.pack()
lbthursday.pack()
thursday.pack()
empty4.pack()

welBtn.pack()
empty5.pack()
  
   
#infinite loop
ws.mainloop()

#--------------------------------------------------------------------------------------------------------------------------------------------------------#
