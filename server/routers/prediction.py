from fastapi import APIRouter, status
from beanie import PydanticObjectId
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from server.models.text_input import TextInput

router = APIRouter()

model_path = "./model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

@router.post("/predict", status_code=status.HTTP_201_CREATED)
async def predict(text_input: TextInput):
    """
    Predict the category of a text

    :param text_input: The text to predict

    :return: {"label": value} where value is the predicted category
    """
    # Tokenize the input text
    inputs = tokenizer(text_input.text, return_tensors="pt", truncation=True, padding=True)

    # Send the input to the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Map the predicted class id to the label
    label_mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    predicted_label = label_mapping[predicted_class_id]

    return {"label": predicted_label}
