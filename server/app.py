# FastAPI app

from fastapi import FastAPI


app = FastAPI()

# Define an endpoint to send request to predict the category of a text
@app.get("/predict")
def predict(text: str):
    # Load the model

    #TODO
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Make the prediction
    outputs = model(**inputs)

    # Get the predicted category
    predicted_category = outputs.argmax().item()

    return {"predicted_category": predicted_category}
    """