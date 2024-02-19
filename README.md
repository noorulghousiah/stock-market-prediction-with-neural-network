# Stock Market Prediction App
This system predicts the stock prices of NVIDIA on Fridays based on the stock prices on the four previous days, that is, on Monday until Thursday.

The Python-based stock price prediction system for NVIDIA follows a sequential process. Initially, sample data spanning 9 weeks is inputted into the system, representing daily stock prices. These values are normalized to fall within the 0 to 1 range.

Before commencing the prediction phase, the system undergoes incremental training to fine-tune the weights between neural network layers. The normalized stock prices from Monday to Thursday serve as inputs for forward propagation, applying set activation functions. Subsequently, the program calculates the predicted stock price for Friday and compares it with the actual value, determining the mean absolute squared loss. This loss is then utilized to update the weights in the neural network. This iterative process is performed for all sample data to obtain optimal weight values.

Once the training phase is complete, the program transitions to prediction mode. Input values, representing stock prices from Monday to Thursday, are fed into the system, triggering forward propagation to produce the predicted stock price for Friday.


This is a group project made by our team: Noorul Ghousiah; Nurain Ainaa Aqilah; Sarah Aishah; Siti Aliaa
