Neural Networks and Deep Learning - Assignment

Name: Pinky Ramnath Mehta
Department: School of Computer Science and Electronic Engineering 

Table of Content:
•	Abstract
•	Background and History
•	Deep Learning Architecture
•	Methods
1.	Logistic Regression Model for Mental Workload Classification
2.	Deep learning Model for Mental Workload Classification
•	Results
1.	Logistic Model accuracy predictions
2.	Deep Learning Model accuracy predictions
•	Conclusion
•	References 
 
## Abstract:

Mental workload classification using deep learning models is an emerging area of research that seeks to improve the accuracy and robustness of workload assessment. Deep learning techniques have shown promising results in various fields, including computer vision and natural language processing, and have recently gained traction in mental workload classification.
Deep learning models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have been applied to classify mental workload using various physiological and behavioural measures. For instance, EEG signals and eye-tracking data can be processed using CNNs to classify different levels of mental workload. 
The report is a demonstration of two forms of mental workload classification, Logistic model and the Deep Learning model using CNN. 
The model one is creating a logistic regression model for the mental workload classification which is a binary classification from scratch using python language with the help of certain in-built libraries like numpy, sklearn, matplotlib. Using these libraries in each function that is defined to build the model and then evaluating the performance of the model with the help of k-fold cross validation and plotting the score on a graph.
The second model is by choosing one of the existing Deep learning models which in this case is Convolutional Neural Network (CNN). The chosen model’s packages and toolboxes are used which is then modified to perform various tasks such as importing the data from the file, training and testing the data, fitting the data into the model, evaluating the model’s performance rate i.e accuracy scores, and finally plotting the acquired scores on a graph and comparing those with the logistic model.

## Background and History:

The classification of mental workload has a long history in psychology and human factors research. The concept of mental workload refers to the amount of cognitive resources required to perform a task. Mental workload can vary depending on the task demands, the individual's skills and abilities, and the environment in which the task is performed.
One of the earliest attempts to classify mental workload was the NASA Task Load Index (TLX), which was developed in the 1980s as a subjective rating scale. The TLX consists of six dimensions: mental demand, physical demand, temporal demand, performance, effort, and frustration. Participants rate their perceived workload on each dimension on a scale from 0 to 100.
The discovery of electroencephalography (EEG) in 1929 by the German psychiatrist Hans Berger was a historical breakthrough providing a new neurologic and psychiatric diagnostic tool at the time, especially considering the lack of all those now available in daily practice (EP, CT, MRI, DSA, etc.) without which the making of neurologic diagnosis and planning neurosurgical operative procedures would now be unconceivable. There are no recent reports on the topic in the Croatian medical literature.
Galvani's accidental discovery of "biological electricity" led to Volta's discovery of the battery (voltaic pile). Using it, Rolando was the first to stimulate cerebral surface. Thus, enabling Fritsch and Hitzig and Ferrier to develop the idea of cerebral localization. 
Berger made the first EEG (electrocorticogram) recording on July 6, 1924, during a neurosurgical operation on a 17-year-old boy, performed by the neurosurgeon Nikolai Guleke. He reported on the topic in 1929, using the terms alpha and beta waves. 
Berger's persistent, hardworking and steady personal style overcame all technical and other obstacles during the experiments. Unfortunately, he gained neither acceptance nor recognition, among his fellow contemporaries from abroad. Political turmoil’s at the dawn of World War II, in the country of Nazi's ideology and finally the outbreak of war, along with the complete ban of any further work on EEG after his forced retirement, led him to an uneasy professional and personal end. In the era when lumbar puncture, pneumoencephalography and ventriculography were the only diagnostic tools to detect and localize "sick sites" in the brain, EEG revolutionized daily neurologic and neurosurgical procedures, and bridged a time period of about 40 years (1930-1970) until the advent of computer tomography. Nowadays its importance is not as great as it was before, but it still has its place in the diagnostic work-up of seizures, brain tumours, degenerative brain changes, and other diseases.
The discovery of electroencephalography was a milestone for the advancement of neuroscience and of neurologic and neurosurgical everyday practice, especially for patients with seizures. The real nature of the disease and its management (anticonvulsants, excision of brain scars, tumours, etc.) were unknown at that time. 
In the 1990s, some researchers also began to explore the use of physiological measures to classify mental workload. These measures include heart rate, skin conductance, and electroencephalography (EEG).
The use of EEG for mental workload classification gained popularity in the early 2000s, with the development of advanced signal processing techniques and machine learning algorithms.
One of the early EEG-based methods for mental workload classification was the spectral power ratio (SPR) method, which compares the power in different frequency bands of the EEG signal to estimate mental workload. 
Another EEG-based method is the event-related desynchronization/synchronization (ERD/ERS) method, which measures changes in the power of EEG oscillations in response to task demands.
In recent years, deep learning approaches have been applied to mental workload classification. These approaches involve training neural networks with large amounts of EEG data to classify mental workload. Deep learning approaches have shown promising results in accurately classifying mental workload in real-time.
The methods used in the study included search through previous reports, bibliographic notes, Internet sources, and analysis of continuous scientific attempts made through centuries to discover the real nature and meaning of electrical activity.
Overall, the history of mental workload classification has seen a progression from subjective rating scales to physiological measures and advanced signal processing techniques. The combination of EEG and deep learning has the potential to provide new insights into mental workload and improve performance in a variety of domains, including healthcare, aviation, and military applications.

## Deep Learning Architecture:
Deep learning architecture refers to the structure and design of artificial neural networks (ANNs) used in deep learning. ANNs are composed of layers of interconnected nodes (also known as neurons) that process and transform data, and deep learning refers to neural networks with multiple layers.
There are several types of deep learning architectures, few includes:
Feedforward Neural Networks: These are the simplest type of neural network, where the data flows only in one direction, from the input layer to the output layer, without any feedback loops.
Convolutional Neural Networks (CNNs): CNNs are commonly used for image and video recognition tasks, where they analyse the data using filters that capture specific features and patterns.
Recurrent Neural Networks (RNNs): RNNs are commonly used for sequential data analysis, such as natural language processing, where they process input data in a sequential manner and retain memory of past inputs.
Autoencoders: Autoencoders are neural networks designed to learn a compressed representation of input data, which can be used for tasks such as data compression and image denoising.
The architecture of a deep learning model is crucial for achieving high performance on a given task. Researchers are constantly developing and refining new architectures to improve the performance of deep learning models on various task.

## Deep learning strategy:
•	AE 	Auto-encoder
•	CNN 	Convolutional neural network
•	Conv 	Convolutional layer
•	DBN 	Deep belief network
•	FC 	Fully connected
•	Hid. 	Hidden layers
•	RBM 	Restricted Boltzmann machine
•	RNN 	Recurrent neural network
•	SVM 	Support vector machine
Activation:
•	ReLU 	Rectified linear unit
•	ELU 	Exponential linear unit
•	L-ReLU 	Leaky rectified linear unit
•	SELU 	Scaled exponential linear unit
•	PReLU 	Parametric ReLU

## Methods:
1.	Logistic Regression model for Mental workload classification.

The first model is logistic regression model for mental workload classification. We have imported the dataset with the help of MATLAB file using the library called SciPy. The same dataset is being used for both training and testing the model for logistic regression.  The features of the dataset is extracted from EEG signals and the respective labels indicates the level of mental workload.

The code is split into five different functions which is used in the logistic regression algorithm. The functions named as Sigmoid function which is the activation function, a feed-forward function for computing the output of the model given the x input, weight and bias, a loss function to measure the error rate between the  predicted and actual labels,  and a backward  propagation  function for updating the weights and bias based on the error and finally calculating the logistic regression with the help of x input, y labelled output, the measuring rate and the maximum number of iterations. The functions with feed_forward which means forward propagation, loss function and backward propagations are stored as y_pred, loss and weight and bias variables respectively and the calculation is completed with the help of for loop for the assigned number of maximum iterations. Once the range of maximum number of iteration calculation is completed it then returns the weight and the bias.

The features and the labels of the dataset is extracted and stored as x_features and y_labels from the file that was imported earlier. Then the shape of the x_features is transposed and reshaped from (62, 512, 360) to (360, 62*512). y_labels are also reshaped and is stored into a new variable called Y_labels. The reshaping of x inputs and y labelled outputs are done for the shape of its rows to match in order to perform the matrix multiplication.

Then, it splits the data into training and testing sets using the train_test_split function from scikit-learn. 
The logistic_regression  function is then called to train the model with learning rate of 0.01 and the maximum number of iterations to be 25. The feed_forward function is then used to predict the labels for the test data. 

The predicted labels are at the range of 0.5 to obtain binary labels and the accuracy of the model is computed using the accuracy score function from scikit-learn.
We are then additionally evaluating the model to predict the performance of the classification model using 5-fold cross-validation with the help of KFold function from scikit-learn.
Overall, this code provides a simple implementation of logistic regression for mental workload classification.

2.	Deep learning model for the mental workload classification.

The second deep learning model for mental workload classification is using Convolutional Neural Network (CNN).

As step one, we are importing libraries which includes sklearn for pre-processing and evaluating the data, for building and training the CNN model we are making use of keras and tensorflow, to get the mean of the scores that we achieve after testing the model we are using numpy and finally to plot the score we are then using matplotlib.

The loading of file is similar to the file importing using scipy.io library that we used in our logistic regression model. 

The x input and y labelled outputs are then extracted from the imported file and stored in as variables data_only and label which then is transposed to perform the matrix multiplication and also that the features and labels are in separate arrays and the samples are in rows.

The data that we have now after performing the required changes is then split into training and testing sets using the train_test_split function from sklearn library.

A Sequential model is defined in Keras, which is a linear stack of layers. The first layer is a 2D convolutional layer with 32 filters, each with a size of 3x3. The activation function used is the Rectified Linear Unit (ReLU). The input shape of the layer is (512, 62, 1), which means that the data has 512-time steps, 62 features, and 1 channel.

A max pooling layer is added with a pool size of 2x2.

Another 2D convolutional layer is added with 64 filters and a size of 3x3, followed by another max pooling layer.

Two more convolutional layers are added with 64 and 50 filters, respectively. The activation function used is ReLU for both layers.

To convert the output from the convolutional layer to a 1D array a Flatten layer is being added.

A size of 64 and 10 neurons is used to create two dense layers. ReLu is the first dense layer used for activation function and the second dense layer uses softmax function to output the probabilities of each class.
The summary is depicted for the model by showing the layers and the number of parameters in each layer.

The compilation of the model is done with the help of kFold cross validation where the number of splits being used is 5, setting the value of Shuffle as True and the random_state as 24.

An empty list for the model accuracy is created at instance and for every train set and test set of the data we have we are then splitting it into 5 folds and compiling with the help of compile function. 

The model is compiled using ‘adam’ as optimiser, loss function as SparseCategoricalCrossentropy and metric accuracy.

The model is trained with training data for 2 epochs and batch size of 20.

After training the model, the model is evaluated for the loss and the accuracy  by passing the x_test and y_test variables and the verbose as 2 since the model has a large number of data.

A mean of all accuracy is listed and then is plotted for the training accuracy over number of epochs using the matplotlib library.

In summary, this python code defines a CNN model for mental workload classification and trains it on a dataset loaded from a MATLAB file using Keras and TensorFlow libraries. It also evaluates the model's accuracy using kfold cross validation and displays a plot of the training accuracy.

## Results:

1.	Logistic Regression model accuracy
The accuracy score for the logistic regression keeps varying between 48% to 50% when the score is calculated for the mentioned number of iterations. 
Reason for score variation - There might be few random fluctuations in the provided data or the algorithm which we have used to fit the model, leading to inconsistent accuracy scores.  So, to get a more reliable estimate of the model's performance we are using kfold cross validation.
When we are evaluating the performance of the model using cross-validation it appears to be almost 50% which indicates that the use of 5-fold cross validation, we can achieve more reliable performance of the model rather than using the usual single train/test split. The benefits of cross-validation is that we prevents overfitting of data and the evaluation is done on multiple sets rather than making use of a single set.

## Accuracy score and Cross-Validation Accuracy Graph: 

2.	Deep Learning model accuracy 
The model’s accuracy score was 87.77% when the evaluation was done with the help of accuracy_score function but then made use of kfold cross validation the score peaked to 96% which indicates that the model performs well with the help of 5-Fold cross validation.
Accuracy Score and Cross-Validation Accuracy Graph:
 
 
## Conclusion:

As we already know that Logistic regression and deep learning models are commonly used for classification tasks. Below are certain comparison between the two models.
1.	Complexity of Model: Logistic model is a simple linear model that maps input features to output classes whereas Deep leaning models have multiple layers of non-linear transformations that will extract complex features from input data. It is clear that deep learning models are more complex than logistic models.
2.	Training and Testing time: The time consumed while training and testing the deep learning model is a lot more than the time consumed for logistic model and this is due to the complexity of the DL model. But this can also depend on the size of the dataset and the hardware that is being used for training and testing purpose.
3.	Performance: The performance of a DL model will often achieve higher accuracy than logistic models, especially with large datasets which is also explained in above graph.

## References:
Deep Learning in EEG: Advance of the Last Ten-Year Critical Period https://ieeexplore.ieee.org/document/9430619
Deep learning for electroencephalogram (EEG) classification tasks
https://iopscience.iop.org/article/10.1088/1741-2552/ab0ab5
Deep Learning Architecture
https://library.samdu.uz/files/e77a6b6ba27540cad1ea4d0096de9195_Deep%20Learning%20Architectures.pdf
Deep Learning model
https://www.tensorflow.org/tutorials/images/cnn





