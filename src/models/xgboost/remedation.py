# remediation.py

def suggest_action(prediction):
    """
    Suggest action based on predicted class
    """
    if prediction == 2:
        return "High priority - investigate immediately"
    elif prediction == 1:
        return "Medium priority - review"
    else:
        return "Low priority - log or ignore"