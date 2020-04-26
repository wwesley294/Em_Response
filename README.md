**EmResponse** extracts and transforms the emergency response data, before feeding it to the ML pipeline to create a classifier that can be implemented in the web app. The web app contains two sections: an user input to initiate the classifier, and visualization on the composition of the emergency response data


## Quick Start & Deployment
`process_data.py` to extract and transform the emergency response data. The data will be stored in a temporary database (or in other forms at the users' choosing).

`train_classifier.py` uses the data from the first step to create a new classifier and preserves it for the next step.

`run.py` deploys generates data visualization and deploys the classifier on the student worksapce via the Udacity IDE platform. However, with a few modification, it can also deploy using Heroku.


## Acknowledgement
Special thanks to Udacity for laying out the steps and to Figure Eight for providing the data.
