import re


def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


def prompt_replace_regex(prompt_style: str, **kwargs) -> str:
    """
    Replaces placeholders in the prompt_style string with corresponding values from kwargs using regex.

    Placeholders are in the format ${key}$.

    Args:
        prompt_style (str): The template string containing placeholders.
        **kwargs: Key-value pairs where keys correspond to placeholder names in the template.

    Returns:
        str: The resulting string with placeholders replaced by their corresponding values.

    Raises:
        KeyError: If a placeholder in the template does not have a corresponding key in kwargs.
    """
    def replacer(match):
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        else:
            raise KeyError(f"Missing value for placeholder: {key}")

    pattern = r'\$\{(\w+)\}\$'
    result = re.sub(pattern, replacer, prompt_style)
    return result


def fill_line_number(source_code):
    code_list = source_code.split('\n')
    code_list = [str(i+1) + " " + line for i, line in enumerate(code_list)]
    return '\n'.join(code_list)
