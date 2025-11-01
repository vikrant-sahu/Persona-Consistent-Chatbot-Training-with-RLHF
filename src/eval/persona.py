# This file contains the implementation for measuring persona consistency in chatbot responses.

import json
import numpy as np

def load_persona_data(persona_file):
    with open(persona_file, 'r') as f:
        return json.load(f)

def calculate_consistency_score(response, persona_traits):
    score = 0
    for trait in persona_traits:
        if trait in response:
            score += 1
    return score / len(persona_traits)

def evaluate_consistency(responses, persona_file):
    persona_data = load_persona_data(persona_file)
    scores = []
    
    for response in responses:
        persona_traits = persona_data.get(response['persona_id'], [])
        score = calculate_consistency_score(response['text'], persona_traits)
        scores.append(score)
    
    average_score = np.mean(scores)
    return average_score

if __name__ == "__main__":
    # Example usage
    responses = [
        {'persona_id': '1', 'text': 'I love hiking and being outdoors.'},
        {'persona_id': '2', 'text': 'I enjoy reading books and writing.'}
    ]
    persona_file = 'data/personas/persona_bank.json'
    consistency_score = evaluate_consistency(responses, persona_file)
    print(f'Average Persona Consistency Score: {consistency_score:.2f}')