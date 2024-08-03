# News Classification App

This is a simple web application that classifies news articles into one of five categories: 
Technology, Business, Sports, Entertainment, and Politics.

## Project Structure

- `app.py`: The main Streamlit application script.
- `models/`: This folder contains the pre-trained models and the TF-IDF vectorizer.
- `text_process.py`: A script for text preprocessing.
- `requirements.txt`: A list of required Python packages.
- `.gitignore`: Specifies which files and directories to ignore in the repository.

## How It Works

1. The user inputs a news article.
2. The article is preprocessed using the `text_preprocessing` function.
3. The preprocessed text is transformed using a TF-IDF vectorizer.
4. The selected machine learning model (RandomForest, DecisionTree, KNN, or Multinomial Naive Bayes) predicts the category of the news article.
5. The predicted category is displayed to the user.

## Models Used

- **RandomForestClassifier**
- **DecisionTreeClassifier**
- **Multinomial Naive Bayes**

## Setup Instructions

1. Clone this repository:
    ```bash
    git clone https://github.com/jack-sparrow4/News_segmentatation_deployment.git
    cd news-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained models and place them in the `models/` directory.

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

5. Open your browser and go to `http://localhost:8501`.

## Requirements

- Python 3.7+
- Streamlit
- Scikit-learn
- Joblib

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to the Scikit-learn library and Streamlit for making this project possible.
