import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def main():
    print("Loading data...")
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "heart.csv")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    print("Preparing data...")
    # Features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    print("Training Random Forest model...")
    # Train Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X, y)
    
    # Save the model
    model_path = os.path.join(script_dir, "model.pkl")
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
        
    print("Model training and saving complete!")

if __name__ == "__main__":
    main()
