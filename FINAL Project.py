import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.models import Sequential

# Load the dataset
data = pd.read_csv("C:\\Users\\Asus\\OneDrive\\Masaüstü\\Python\\coin_shots.csv", delimiter=";")

# Print column headers
print(data.columns)

# Define labels for Heads and Tails
Heads = 1
Tails = 0

# Label the data
data["Out_come"] = data["Out_come"].replace("Heads", Heads)
data["Out_come"] = data["Out_come"].replace("Tails", Tails)

# Print the labeled dataset
print(data.head())

# Map the orientation of the coin
data["Orientation"] = data["Orientation"].replace("heads up", 0)
data["Orientation"] = data["Orientation"].replace("vertical", 1)
data["Orientation"] = data["Orientation"].replace("tails up", 2)

# Convert distance from origin to float
data["Distance_From_Origin"] = data["Distance_From_Origin"].astype(float)

# Print the preprocessed dataset
print(data.head())

# Define Neural Network architecture
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(4,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile and train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(data[["Orientation", "Distance_From_Origin"]], data["Out_come"], epochs=100)

# Evaluate the model's performance
print(model.evaluate(data[["Orientation", "Distance_From_Origin"]], data["Out_come"]))

# Make predictions for a new coin toss
new_data = pd.DataFrame({
    "Orientation": 0,
    "Distance_From_Origin": 0.5,
})

# Print the prediction
prediction = model.predict(new_data)
print(prediction)
