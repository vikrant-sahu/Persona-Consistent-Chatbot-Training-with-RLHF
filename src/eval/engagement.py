def calculate_engagement_metrics(conversations):
    """
    Calculate engagement metrics for a list of conversations.

    Parameters:
    conversations (list): A list of conversation dictionaries, where each dictionary contains
                          'user_message' and 'bot_response'.

    Returns:
    dict: A dictionary containing engagement metrics.
    """
    total_messages = len(conversations)
    total_user_messages = sum(1 for conv in conversations if conv['user_message'])
    total_bot_responses = sum(1 for conv in conversations if conv['bot_response'])

    # Example metrics
    user_engagement_ratio = total_user_messages / total_messages if total_messages > 0 else 0
    bot_engagement_ratio = total_bot_responses / total_messages if total_messages > 0 else 0

    metrics = {
        'total_messages': total_messages,
        'total_user_messages': total_user_messages,
        'total_bot_responses': total_bot_responses,
        'user_engagement_ratio': user_engagement_ratio,
        'bot_engagement_ratio': bot_engagement_ratio
    }

    return metrics


def evaluate_engagement(conversations):
    """
    Evaluate engagement based on conversation data.

    Parameters:
    conversations (list): A list of conversation dictionaries.

    Returns:
    None: Prints the engagement metrics.
    """
    metrics = calculate_engagement_metrics(conversations)
    print("Engagement Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")