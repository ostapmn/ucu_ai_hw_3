def head_tail(tokens, body, head):
    if len(tokens) <= body:
        return list(tokens)
    return list(tokens[:head]) + list(tokens[len(tokens) - (body - head) :])
