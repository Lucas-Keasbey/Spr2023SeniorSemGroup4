# Spr2023SeniorSemGroup4

Group Members and Group Number:  Jacob Schnitker, Walker Hayes, Colton Glidewell, Lucas Keasby, Group 8

Project Title: Computationally Categorizing Clothing


Project Executive Summary:
	This project is a preliminary work on Machine Learning using the PyTorch development environment in the domain of image classification. We will be performing a parametric comparison of multiple data training techniques, plotting loss vs both training rate and epochs to determine the optimal or near optimal performance for this model.

Problem Definition:
	The problem we are attempting to address is the classification of various articles of clothing. The end user could be someone who is trying to organize their online clothes store website. Rather than put each product in a category individually, the user would be able to upload all of the pictures of their products into our software and have the program separate them into individual categories. A profile for this person could be a mid 40s business owner.

Proposed Solution:
This project is significant because AI is more relevant than ever today, and one of the most important requirements of a software engineer is staying up to date with the latest technology. This project will help us familiarize ourselves with AI, how it works, and how to implement it. Android studio is also another important component of this project. We will learn how to write programs that will be utilized by one of the biggest operating systems that is not PC based. Our proposed solution to this project is to use the techniques and skills we learned in the software engineering course and over the entirety of our major. We will achieve this by choosing a software engineering development style (I.E agile) and through the use of a VCS. We will use Android Studio to construct the app and the Python library, PyTorch, to complete the training model. Visualization such as graphs detailing different learning rates and epochs will come form the Python library Matplotlib. All of us will be working equally hard to complete the project, though the member with the GPU will be working more personally with the training model as they will be able to test parameters with more efficiency. Our group will be meeting every time the class is not meeting to go over and discuss: our progress, next steps, etc. Besides that, we will also utilize online messaging such as Discord to give regular updates on what we do outside of those class periods.
We will be using three different models in our code. The first Model will be a Convolutional Neural Network (CNN) provided for us, the second will be a Linear model that uses only linear layers, and the third will be a CNN with 4 convolution layers and 1 linear layer. These models will be compared in terms of both loss and time efficiency.

Timeline:

	February 27th: Skeleton for the android app and basic training model created
At this point, we will be spending most of our time creating the VCS and creating the layout for the app. We will not be able to put much effort into the training model beyond using the basic one we presented ourselves.

	March 13th: Find the ideal training rate/number of epochs
With the training model changed to use the Fashion dataset, we will be spending this time finding what details a good learning rate and number of epochs. This data will go into graphs for comparisons.

	March 27th: Fully train the model with different training rates and epochs
			Sub goal: begin developing linear and CNN models
Once the optimal parameters have been chosen, the next two weeks will be spent training the model as much as we can. Since it can only be efficiently trained on one computer, the rest of us will work to begin creating the linear and CNN models.

	April 10th: Have the first model trained and in the app.
By mid-April, the first model should be completely trained and put into the app. At this point, we hope to have the basic project completely finished.

	April 24th: Complete the linear and CNN model
The rest of the time will be spent finishing the documentation, cleaning up the code and submitting our work. Ideally, we will also import the two additional models into the application.
	

Reflection:
	This qualifies as a capstone project because we are simulating what it's like to be a software engineer and we are bringing together and applying all the knowledge we have learned over our college career. This is because we are using many of the same techniques and tools that one would use in a real life job setting. For example, working as a team to develop an application, using GitHub, etc. This project will hone in our skills as a software developer as we will be making an application with some actual significance and with real life applications. As we are in a school setting, there is a safety net in the form of our professor and colleagues. If we are truly stuck, there are many people we can go to for help which is a valuable skill to have in and of itself.
	
	
How to run the code:
	1. Simply download this repo and put the folder where ever you want it.
	2. Python Code:
		2a. Pick you favorite python IDE of choice (However Visiual studio is prefered) and use the repo itself as the root folder for the project
		(WARNING: Spyder and Visual Studio use different methods of using files strings so the code may not run due to a few dots before the file its trying to 		access)
		2b. To train the a model, run AllModelTrainCode with args model type (Basic,Linear,CNN), num_epochs, and learning rate.
		2c. To test a model, run AllModelTestCode, and speficy what model you want to test after running the program
		2d. To use your trained model in the app, first run the ModelAppConverter code and speficy what model you want to convert. Once complete, move the model 			into the assets folder in android studio.
		2e. DO NOT RUN NNSkeletonCode or BasicModelTraining!!!, these are depreicated and kept around for reference
	3. Android stuido app:
		3a. Run android studo and choose the root project folder as Application, this is very important as it will not work otherwise.
		3b. Run the app and choose any image you would like and select what model you waant to use, press classify and obsever the prediction
		
	
	
	
	
	
	
	
	
	
	
	
