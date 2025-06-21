import re

# Original MATLAB expression
matlab_expr = """-29.868689780242356 + 23.80563018566768*y - 4.753125994591446*x + 9.459754038108372*y^2 + 2.4372132851840154*x*y - 4.757502044306331*x^2 - 7.2568449649792095*y^3 + 0.5270410399146362*x*y^2 + 1.0832150213439806*x^2*y + 0.4849555917663688*x^3"""

def append_after(str, i, text):
    prev = str[:i]
    after = str[i:]
    return prev + text + after

def expression_convert(matlab_expr):
    # Replace '^' with '**' for power notation
    matlab_expr = re.sub(r'\^', '**', matlab_expr)

    # Replace 'x' with 'y[0]' ensuring no unintended replacements
    matlab_expr = re.sub(r'\bx\b', 'y[0]', matlab_expr)


    matlab_expr = append_after(matlab_expr, len(matlab_expr), "$")

    i = 0
    while matlab_expr[i] != '$':
        if matlab_expr[i] == 'y' and matlab_expr[i + 1] != "[":
            matlab_expr = append_after(matlab_expr, i + 1, "[1]")
            i = i + 3
        i = i + 1

    matlab_expr = matlab_expr[:-1]
    return matlab_expr

py_expr = expression_convert(matlab_expr)
print(py_expr)


# -29.868689780242356 + 23.80563018566768*y - 4.753125994591446*x + 9.459754038108372*y^2 + 2.4372132851840154*x*y - 4.757502044306331*x^2 - 7.2568449649792095*y^3 + 0.5270410399146362*x*y^2 + 1.0832150213439806*x^2*y + 0.4849555917663688*x^3