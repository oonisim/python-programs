# 'import ...' is for absolute import only
# For relative import, must be 'from ... import ...'
# See https://www.python.org/dev/peps/pep-0328/#guido-s-decision
from . other import use_me
if __name__ == "__main__":
    use_me()
