# End_To_End-Movie_Review_Analyze_Using_Docker
Building an end to end machine learning pipeline and then finally dockerize in order to package the source code that can be easily shared with other teams and all they have to do is run that docker image.


#### Let us first outline the steps that we will be performing

* 1. We will be building a movie review analyzer model that takes in movie reviews as input and outputs whether its a positive or negative sentiment.
* 2. First we will create a train.py file that we will use to load the data, clean and preprocess it and finally train a simple logistic regression model on top of it.
* 3. Next we will create an inference.py file that we will use to load our trained model and make predictions on any testing set.
* 4. Create a docker file that will contain instructions on how to build the docker image(for now just think of this as a set of instructions on a file. I will explain docker when we start that section later in the article).
* 5. Using the docker file build your docker image(this will internally run our train.py file).
* 6. Run the docker image(this will run the inference.py file) with any test set that you want from your local(this was the meaningful part I was talking about).
 
  #### The next step is to build a docker image using this docker file.
  * Please ensure docker desktop is up and running before you run docker commands.

    ![image](https://github.com/user-attachments/assets/fad2e3ea-0446-4398-b3e4-31f23c3cd907)
