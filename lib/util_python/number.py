"""Module for number handling utilities
"""


def is_int_string(x: str):
    """Check if a string is integer or float
    123.0 and 123 are both integer
    """
    assert isinstance(x, str)
    if '.' in x:
        i_f = x.split('.')
        i = i_f[0]         # integer part
        f = i_f[1]         # fraction part
        return (
            # Integer part
            (
                    i.lstrip('-').isdigit() or         # 123, -123
                    len(i.lstrip('-')) == 0            # i part of ".0" or "-.0"
            )
            and
            # Fraction part
            (
                    (f.isdigit() and int(f) == 0) or   # 123.0, 123.00
                    len(f) == 0                        # 123.
            )
        )
    else:
        return x.lstrip('-').isdigit()


