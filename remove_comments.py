import re
import sys

def remove_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove docstrings (triple-quoted strings)
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    
    # Process line by line to remove single-line comments
    lines = content.split('\n')
    result_lines = []
    
    for line in lines:
        # Remove comments that start with #
        if '#' in line:
            code_part = line.split('#')[0]
            # Keep the line if there's code before the comment
            if code_part.strip():
                result_lines.append(code_part.rstrip())
            # Skip lines that are only comments
        else:
            result_lines.append(line)
    
    # Join lines back together
    result = '\n'.join(result_lines)
    
    # Remove extra blank lines (more than 2 consecutive)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        remove_comments(sys.argv[1])
    else:
        print("Please provide a file path")
