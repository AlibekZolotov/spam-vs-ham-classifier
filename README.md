# spam-vs-ham-classifier
A simple Python script that classifies messages as spam or ham (not spam) using CountVectorizer for text feature extraction and Multinomial Naive Bayes for classification. 
Features

Loads a dataset (combined_dataset.csv) containing messages labeled as spam or ham.

Converts text messages into numeric features using Bag-of-Words (CountVectorizer).

Trains a Naive Bayes classifier to predict spam messages.

Evaluates model performance with accuracy, confusion matrix, and classification report.

Interactive terminal interface: type a message to check if it is spam or ham.

Dataset

The script expects a CSV file named combined_dataset.csv with the following columns:

Column	Description
target	Message type (ham or spam)
text	The message content
Installation

Make sure you have Python 3.13+ installed and install the required packages:

pip install pandas scikit-learn numpy
Usage

Run the classifier script:

python spam_classifier.py

You can then enter any message to see whether it is classified as spam or ham. Type quit to exit the program.

Example Output
Enter a message (or type 'quit' to exit): Hi
✅ This message is HAM (not spam)!

Enter a message (or type 'quit' to exit): WIN a free iPhone now!
🚨 This message is SPAM!
Author

Alibek Berik
November 2025

License

This project is open source and free to use.
