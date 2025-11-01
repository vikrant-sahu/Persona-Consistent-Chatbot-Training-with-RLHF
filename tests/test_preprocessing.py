import pytest
from src.data.preprocessor import preprocess_data

def test_preprocess_data():
    # Test case for valid input
    input_data = [
        {"text": "Hello, how are you?", "persona": "friendly"},
        {"text": "I'm doing well, thank you!", "persona": "friendly"}
    ]
    expected_output = [
        {"text": "hello how are you", "persona": "friendly"},
        {"text": "im doing well thank you", "persona": "friendly"}
    ]
    output_data = preprocess_data(input_data)
    assert output_data == expected_output

def test_preprocess_data_empty_input():
    # Test case for empty input
    input_data = []
    expected_output = []
    output_data = preprocess_data(input_data)
    assert output_data == expected_output

def test_preprocess_data_invalid_input():
    # Test case for invalid input
    input_data = [
        {"text": None, "persona": "friendly"},
        {"text": "I'm doing well, thank you!", "persona": "friendly"}
    ]
    with pytest.raises(ValueError):
        preprocess_data(input_data)