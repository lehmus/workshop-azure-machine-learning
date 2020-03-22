## Lab 2: running experiments ##
Enhance the model creation process by tracking your experiments and monitoring run metrics. In this lab, learn how to add logging code to your training script, submit an experiment run, monitor that run, and inspect the results in Azure Machine Learning. We are going to perform the following steps in the lab:
* Understand the "normal" train script
* Run the train script local
* Refactor the train script to log metrics to Azure Machine Learning
* Submit an experiment 
* Monitor the run
* Inpect the results in Azure Machine Learning

![](C:/Users/mideboer.EUROPE/Documents/GitHub/aml-mlops-workshop/Images/02_experiments.PNG)

# Pre-requirements #
1. Familiarize yourself with the concept of Azure Machine Learning by going though the [Introduction](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/Powerpoints/Module%201%20-%20Introduction.pptx)
2. Familiarize yourself with the concept of experiments by going though [Experiments](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/Powerpoints/Module%202%20-%20Experiments.pptx)
3. Read the documentation on [Azure Machile Learning architecture](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture)
4. Finished the setup file [01_setup](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/01_setup.md)


# Understand the non-azure / open source ml model code #
We first start with understanding the training script. The training script is an open source ML model code from https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html.  The dataset used in this example is the 20 newsgroups dataset. It will be automatically downloaded, then cached. The newsgroup datasets contains text documents that are classified into 20 categories.

Open the train.py document to inspect the code.
    
The first step in the code is to load the dataset from the 20 newsgroup dataset. In this example we are only going to use a subset of the categories. Please state the catogories we are going to use:

...

The second step is to extract the features from the text. We do this with a sparse vecorizer. We also clean the data a bit. What is the operation that we do on the data to clean the text?

...

After we have reshaped our data and made sure the feature names are in the right place, we are going to define the algorithm to fit the model. This step is defining the benchmark. We fit the data and make predictions on the test set. To validate out model we need a metric to score the model. There are many metrics we can use. Define in the code the metric that you want to use to validate your model and make sure the print statement will output your metric. (Note: you can define multiple scores if you want. If so, make sure to return these scores.)

...


The last step is to define tha algoritms that we want to fit over our data. In this example we are using 1 classification algoritm to fit the data. We keep track of the metrics of all algoritms, so we can compare the performance and pick the model. Look at the code and whrite down the algoritm that we are going to test.

...

# Run the training locally #
Just to check, we are now going to train the script locally without using Azure ML. 
1. Execute the script `code/explore/train_1model.py`

#  Run the code via Azure ML #
Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Part 1: Refactor the code to capture run metrics in code/explore/train.py
We can caputure the results from our run and log the result to Azure Machine Learning. This way we can keep track of the performance of our models while we are experimenting with different models, parameters, data transformations or feature selections. We can specify for ourselfves what is important to track and log number, graphs and tables to Azure ML, including confusion matrices from SKlearn. For a full overview check the [avaiable metrics to track](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments#available-metrics-to-track)


1. Get the run context
    First step is o get the run context. A run represents a single trial of an experiment. Runs are used to monitor the asynchronous execution of a trial, log metrics and store output of the trial, and to analyze results and access artifacts generated by the trial. The get_context() statement returns current service context. We use this method to retrieve the current service context for logging metrics and uploading files.
    ```python
    from azureml.core import Run
    run = Run.get_context()
    ```

2. Log the metric in the run
     Logging a metric to a run causes that metric to be stored in the run record in the experiment. You can log the same metric multiple times within a run, the result being considered a vector of that metric. In this lab, we are going to log the accuracy of the model. Later, in future labs we include more advanced logging like graphs and confusion matrices.

    ```python
    run.log("accuracy", float(score))
    ```

3. upload the .pkl file to the output folder
    The pkl file contains the model we have trained. We want to upload the model to the Azure Machine Learning Service. We need to create an outputs folder if one is not present yet where we can save the model at the top of the file.

    ```python
    import os

    # Define ouputs folder
    OUTPUTSFOLDER = "outputs"

    # create outputs folder if not exists
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)
    ```

    Next, at the end of the file, once we have trainded our model, we want to save that model in the outputs folder we have just created, by the following code:

    ```python
    from sklearn.externals import joblib

    # save .pkl file
    model_name = "model" + ".pkl"
    filename = os.path.join(OUTPUTSFOLDER, model_name)
    joblib.dump(value=clf, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)
    ```

4. close the run

    ```python
    run.complete()
    ```

5. Execute the refactored script `code/explore/train_1model.py`
    As an output you should get the following:
    ```python
    Attempted to log scalar metric accuracy:
    0.7834441980783444
    Attempted to track file modelRandom forest.pkl at outputs/modelRandom forest.pkl
    Accuracy  0.783
    ```
Since we did not submit the run to Azue ML, the log metrics and model file are not logged yet. To log the metrics and model to Azure ML, we need to submit an experiment to the service. We will do this is a seprated python file `code/explore/train_1model.py` in the next part of this lab

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/explore/train_1model.py)

## Part 2: Create the train_1model_submit.py file
In this part, we are going to create the submit file.
1. Load required Azureml libraries

    ```python
    import os
    from azureml.core import Workspace, Experiment
    from azureml.train.estimator import Estimator
    from azureml.core.authentication import AzureCliAuthentication
    ```

2. Load Azure ML workspace form config file
    In order to submit the experiment to Azure ML, we need to get the workspace in our Azure account where we want to submit our run. You can hard-code your credentials here, but it is advised to use a config file. When we create the workspace by running the `infrastructure\create_mlworkspace.py`, we created a config file in `.azureml/config.json`. By using this file and the Azure Cli authentification, we can easily retrieve the required workspace.

    ```python
    # load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())
    ```

3. Create an extimator to define the run configuration
    Before we can sumbit the run, we need to creat the run estimator. The Estimator class wraps run configuration information to help simplify the tasks of specifying how a script is executed. It supports single-node as well as multi-node execution. Running the estimator produces a model in the output directory specified in your training script. As an entry script, we are going to run our `train_1model.py` file. As compute target, we specify 'local'. This means that we are going to execute the script on our local computers. This way, everything stays on you local computur (including data) and only the specified logs will be send to Azure ML. In future labs, we will see how we can use remote compute to execute runs remotely. In de `conda_dependencies.yml`, we specify all conda and pip packages our `train_1model.py ` script in depended on. As we know that locally we have all the packages already installed, we will not use docker at this point. Using dokcer come in handy when we want to excecute our script remotely. We will see this in later labs.

    ```python
    # Define Run Configuration
    est = Estimator(
        entry_script='train_1model.py',
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target='local',
        conda_dependencies_file=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../',
            'conda_dependencies.yml'
        ),
        use_docker=False
    )
    ```

4. Define the ML experiment
    An Experiment is a class acts as a container of trials that represent multiple model runs. Within an experiment we can easily compare different runs of the same script with, for example, different models,parameters or data.

    ```python
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train_15models")
    ```

5. Submit the experiment
    We are now ready to submit the experiment:

    ```python
    # Submit experiment run, if compute is idle, this may take some time')
    run = experiment.submit(est)

    # wait for run completion of the run, while showing the logs
    run.wait_for_completion(show_output=True)
    ```


6. Go to the portal to inspect the run history

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/explore/train_1model_submit.py)
