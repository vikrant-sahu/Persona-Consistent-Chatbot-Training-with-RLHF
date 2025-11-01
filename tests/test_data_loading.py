import os
import json
import pytest
from src.data.loader import load_personachat, load_blended_skill_talk, load_custom_personas

def test_load_personachat():
    data = load_personachat()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(item, dict) for item in data)

def test_load_blended_skill_talk():
    data = load_blended_skill_talk()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(item, dict) for item in data)

def test_load_custom_personas():
    data = load_custom_personas()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(item, dict) for item in data)

def test_data_integrity():
    personachat_data = load_personachat()
    blended_skill_data = load_blended_skill_talk()
    custom_personas_data = load_custom_personas()

    assert len(personachat_data) > 0
    assert len(blended_skill_data) > 0
    assert len(custom_personas_data) > 0

    # Check for required fields in PersonaChat
    required_fields = ['persona', 'utterance']
    for item in personachat_data:
        assert all(field in item for field in required_fields)

    # Check for required fields in Blended Skill Talk
    required_fields_bst = ['context', 'response']
    for item in blended_skill_data:
        assert all(field in item for field in required_fields_bst)

    # Check for required fields in Custom Personas
    required_fields_custom = ['name', 'traits']
    for item in custom_personas_data:
        assert all(field in item for field in required_fields_custom)