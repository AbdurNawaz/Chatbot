# Chatbot
An LSTM based Machine learning model that can chat with you.
This model has been trained on [conversion.json](conversation.json), which contains a normal conversation between two people.<br/>
Since this model has been trained on a very small data, sometimes it makes mistakes.

### Architecture
The preprocessing done by seperating the data into question and answer part and then using [word2vec](https://github.com/jhlau/doc2vec) word embeddings to convert each word into vector and then process it.<br/>
This model is a 3 layer LSTM network having **tanh** activation function on all of them and then using a **Cosine Proximity** loss function to calculate the loss.<br/>
[]('arch.png')
