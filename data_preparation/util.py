def snake_to_camel(s: str) -> str:
    parts = s.split('_')
    return parts[0] + "".join([word.title() for word in parts])
