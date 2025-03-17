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
  Please ensure docker desktop is up and running before you run docker commands.

  *  The first command is the docker build command using **docker build -t movie_review_analyzer** . command 

    ![image](https://github.com/user-attachments/assets/fad2e3ea-0446-4398-b3e4-31f23c3cd907)

  * Once the build command is successful we can see the list of docker images created by running the **docker images** command.
      ![image](https://github.com/user-attachments/assets/187253c2-7e65-466d-b3a3-9576cb6b82aa)
  * Now the next step is to run this image and we do that using the **docker run** followed by image_ID command.
    ![image](https://github.com/user-attachments/assets/b89c6892-2fb2-4c22-8615-854d3d1afc22)
    With docker run we specify the image id that we want to convert into a container and also specify that we want to execute it in detach mode(which basically means it will run in the background and we can do other things like execute bash in it). For example we can go inside the container and see what we have. We can do that using the docker exec command.
    ![image](https://github.com/user-attachments/assets/d60aaea9-a4e8-486d-96e1-3d9c593da2d8)
  
* Let's now just write exit and move out the container so that we can perform the next steps. We will now run the inference.py script on the internal test set.
  ![image](https://github.com/user-attachments/assets/208428a8-dec0-4f19-b638-f637af9cdadb)
  We run the same command as previous with few modifications

* -e COMPUTE_METRICS — We want to change the environment variable so that we can get the model performance metrics. We do this because we are testing on the internal test set.
* python src/inference.py — We want to run inference.py script inside the container.

  Finally you can stop your running container by using the **docker container stop {container_id}**.



