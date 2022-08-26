__environment = None
__project_id = None


def init(environment: str, project_id: str):
    # Access outer scope module variables
    global __environment, __project_id
    __environment = environment
    __project_id = project_id


def get():
    return __environment, __project_id
