import pytest
from src.inference.chatbot import Chatbot

@pytest.fixture
def chatbot():
    return Chatbot()

def test_chatbot_initialization(chatbot):
    assert chatbot is not None

def test_chatbot_response(chatbot):
    response = chatbot.get_response("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_chatbot_persona_control(chatbot):
    chatbot.set_persona("friendly")
    response = chatbot.get_response("Tell me a joke.")
    assert "joke" in response.lower()

def test_batch_inference(chatbot):
    inputs = ["Hello!", "What is your name?", "Tell me a story."]
    responses = chatbot.batch_inference(inputs)
    assert len(responses) == len(inputs)
    for response in responses:
        assert isinstance(response, str)
        assert len(response) > 0