import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

def train_model(phase):
    data = pd.read_csv('ball_by_ball_ipl.csv')
    # Considering 2nd innings data
    df = data.loc[data['Innings'] == 2]
    df = df.drop(columns = ['Unnamed: 0'])

    inf_features = ['Over', 'Ball', 'Batter Runs', 'Extra Runs', 'Runs From Ball', 'Ball rebowled', 'Runs to Get', 'Valid Ball']

    features = ['Runs From Ball', 'Innings Runs', 'Innings Wickets',
        'Target Score', 'Balls Remaining', 'Total Batter Runs',
        'Total Non Striker Runs', 'Batter Balls Faced',
        'Non Striker Balls Faced']

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # df = innings2
    phases = ['Powerplay', 'Middle Overs', 'Final Overs']
    if phase == 'Powerplay':
            data = df[df['Balls Remaining'] > 84]
    elif phase == 'Middle Overs':
        data = df[(df['Balls Remaining'] > 30) & (df['Balls Remaining'] <= 84)]
    else:
        data = df[df['Balls Remaining'] <= 30]

    # Step 2: Set the cutoff date
    cutoff_date = '2018-01-01'

    # Step 3: Split the data into training and test sets based on the cutoff date
    train_data = data[data['Date'] < cutoff_date]
    test_data = data[data['Date'] >= cutoff_date]


    # Step 4: Select the relevant features for X and the target variable for y
    X_train = train_data[features]
    y_train = train_data['Chased Successfully']
    X_test = test_data[features]
    y_test = test_data['Chased Successfully']

    # Step 5: Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression(C=1)
    model.fit(X_train, y_train)

    # Evaluate the model on the testing set
    LR_score = model.score(X_test, y_test)

    # Print accuracy of the model
    return f"Accuracy of {phase} Classifier: {LR_score}"

    
if __name__ == "__main__":
    st.title("IPL Prediction by Phase")

    left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:
    # left_column.button('Press me!')

    # Or even better, call Streamlit functions inside a "with" block:
    with right_column:
        chosen = st.radio(
            'Sorting hat',
            ("Powerplay", "Middle Overs", "Final Overs"))
        st.write(f"Predict with {chosen} data?")


    
    # Button
    if left_column.button("Train"):
        left_column.write("Button clicked!")
        # Place your code here
        output = train_model(chosen)
        st.write(output)