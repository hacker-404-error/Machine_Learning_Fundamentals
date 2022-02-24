# Machine_Learning_Fundamentals

## What Is Machine Learning ?

<p>
Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

`Supervised learning and Unsupervised learning`.
</p>

## Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into `regression` and `classification` problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 

Example 1:

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Example 2:

(a) `Regression` - Given a picture of a person, we have to predict their age on the basis of the given picture

(b) `Classification` - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

![Classification And regression](https://media.geeksforgeeks.org/wp-content/uploads/classification_regression.png)

## Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by `clustering` the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

![Unsupervised_Learning](https://media.geeksforgeeks.org/wp-content/uploads/unsupervised_learning-.png)


Example:

`Clustering`: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

`High Dimension Visualization`: Use the computer to help us visualize high dimension data.

`Generative Models`: After a model captures the probability distribution of your input data, it will be able to generate more data. This can be very useful to make your classifier more robust.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect).


## Important Terms

#### Model
A model is a specific representation learned from data by applying some machine learning algorithm. A model is also called hypothesis.

#### Feature
A feature is an individual measurable property of our data. A set of numeric features can be conveniently described by a feature vector. Feature vectors are fed as input to the model. For example, in order to predict a fruit, there may be features like color, smell, taste, etc.
Note: Choosing informative, discriminating and independent features is a crucial step for effective algorithms. We generally employ a feature extractor to extract the relevant features from the raw data.

#### Target (Label)
A target variable or label is the value to be predicted by our model. For the fruit example discussed in the features section, the label with each set of input would be the name of the fruit like apple, orange, banana, etc.

#### Training
The idea is to give a set of inputs(features) and it’s expected outputs(labels), so after training, we will have a model (hypothesis) that will then map new data to one of the categories trained on.

#### Prediction
Once our model is ready, it can be fed a set of inputs to which it will provide a predicted output(label).

![Training](https://media.geeksforgeeks.org/wp-content/uploads/training.png)

# Python libraries that used in Machine Learning are: 

#### `Numpy` [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hacker-404-error/Machine_Learning_Fundamentals/blob/master/NumPy/Numpy.ipynb)
#### `Scipy`
#### `Scikit-learn`
#### `Theano`
#### `TensorFlow`
#### `Keras`
#### `PyTorch`
#### `Pandas` [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hacker-404-error/Machine_Learning_Fundamentals/blob/master/Pandas/Pandas.ipynb)
#### `Matplotlib` 