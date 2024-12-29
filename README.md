# grad-project-integration

## Installation
Follow the following steps to run the server:

1. You must have Python installed on your system (any Python version 3.x)
2. Clone the repo: `git clone https://github.com/gangsterv/grad-project-integration`
3. `cd` into the repo folder
4. Create a new Python virtual environment using the command: `python -m venv env`
5. Activate the virtual environment by running the script: `.\env\Scripts\Activate.ps1`.
    - This works for Windwos. For other OS, please run the compatible script inside Scripts folder
6. Install the required packages: `pip install -r requirements.txt`
7. Launch the server: `python app.py`

## Usage
Now the server is deployed and ready to use. 
- Base endpoint: `http://localhost:5000`
- To check the server health, send a `GET` request to the default page: `http://localhost:5000/`
- To get predictions, send a `POST` request to the route: `http://localhost:5000/predict`
    - Expects JSON data with two keys:
        - `word`: the word to predict
        - `audio`: the audio wave object

You can follow the example in `example.ipynb` notebook as a demo on usage.