"""
Spreadtable processing
Parse a given CSV file and evaluate each cell by these rules:
1. Each cell is an expression in postfix notation. Please refer to the wikipedia page for a full description.
2. Each number or operation will always be separated by one or more spaces.
3. A cell can refer to another cell, via the LETTER NUMBER notation
   (A2, B4, etc - letters refer to columns, numbers to rows).
4. Support the basic arithmetic operators +, -, *, /

The output will be a CSV file of the same dimensions, where each cell is evaluated to its final value.
If any cell is an invalid expression, then for that cell only print #ERR.

For example, the following CSV input:
```
10, 1 3 +, 2 3 -
b1 b2 *, a1, b1 a2 / c1 +
+, 1 2 3, c3
```

Might output something like this:
```
10,4,-1
40,10,-0.9
#ERR,#ERR,#ERR
"""
import logging
import sys
from itertools import (
    product
)
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Callable,
    Iterable,
    Any,
    Optional,
    Union,
)

import networkx as nx
import numpy as np
import pandas as pd
from utility import (
    CSV_FIELD_DELIMITER,
    OS_EOL_CHARACTER,
    is_number
)
from networkx_utility import (
    list_reverse_topological_sorted_paths_in_graph,
)
TOKEN_TYPE_TERMINAL = 1
TOKEN_TYPE_OPERATOR = 2
TOKEN_TYPE_REFERENCE = 3
TOKEN_TYPE_INVAID = -1
OPERATORS = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "%": np.mod,
}


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_table(path_to_file: str) -> pd.DataFrame:
    """Load table structure as a dataframe
    Prerequisite:
        The csv has the header as the first line and starts with 'a'.
        It is difficult to identify if CSV file has a header for all string table.
        csv.Sniffer().has_header(f.read(1024)) cannot determine.

    Args:
        path_to_file: path to the file
    Returns: dataframe
    """
    df: pd.DataFrame = pd.read_csv(
        path_to_file,
        header=0,
        dtype=str,
        engine="c",
    )
    assert df.columns[0] == 'a', "CSV does not have a header line starting with 'a'."
    # Reindex column from 1
    df.index += 1
    return df


def parse_table_into_graph(table: pd.DataFrame) -> nx.DiGraph:
    """Generate a graph representing the spredtable in which cells are nodes.
    Args:
        table: table structure as pandas dataframe
    Returns: directed graph as NetworkX.DiGraph
    """
    assert isinstance(table, pd.DataFrame) and table.count().sum() > 0, \
        "source table or dataframe is empty"

    graph: nx.DiGraph = nx.DiGraph()

    # --------------------------------------------------------------------------------
    # All the cell names e.g. {"a1", "a2", "b1", "b2"}.
    # If dataframe does not have columns, set them as "a, b, ..., z" where zz is max.
    # --------------------------------------------------------------------------------
    cell_names: Set = {
        f"{column}{row}"
        for column, row in product(table.columns.tolist(), table.index.tolist())
    }

    for column in table.columns:
        for row in table.index:
            current_cell_name: str = f"{column}{row}"
            
            # --------------------------------------------------------------------------------
            # Expression in the cell e.g. "1 3 * 2 /"
            # --------------------------------------------------------------------------------
            cell: str = table.at[row, column]
            expression = " ".join(cell.split())

            # --------------------------------------------------------------------------------
            # References to other cells if any e.g. expression = "1 b1 + c2 *"
            # --------------------------------------------------------------------------------
            references = set(expression.split()) & cell_names
            
            # --------------------------------------------------------------------------------
            # Self refence expression is invalid e.g. "a1 1 +" in the cell a1
            # --------------------------------------------------------------------------------
            value: float = np.NaN
            validity: bool = True
            if current_cell_name in references:
                value = np.NaN
                validity = False
                logger.error("cell[%s] expression has self reference", current_cell_name, expression)
            
            # --------------------------------------------------------------------------------
            # Edges from the current node to the cells referenced
            # --------------------------------------------------------------------------------
            edges = product([current_cell_name], references)
            graph.add_edges_from(edges)
            
            # --------------------------------------------------------------------------------
            # Node representing the current cell
            # --------------------------------------------------------------------------------
            graph.add_node(
                current_cell_name,
                valid=validity,
                expression=expression,
                value=value,
            )

    return graph


def evaluate_token(
    token: str, 
    references: Iterable[str]
) -> Tuple[int, Union[float, Callable, str]]:
    """Evaluate individual token
    Token is an element in an expression e.g. "1 a2 * 2 +".
    
    Args:
        token: Token to evaluate
        references: List of references which is cell names e.g. "a1"
    """
    token = token.strip()

    result, number = is_number(token)
    if result:
        return TOKEN_TYPE_TERMINAL, number

    if token in OPERATORS.keys():
        return TOKEN_TYPE_OPERATOR, OPERATORS[token]
    
    if token in references:
        return TOKEN_TYPE_REFERENCE, token

    logger.error("evaluate_token(): invalid token [%s]", token)
    return TOKEN_TYPE_INVAID, token


def evaluate_expression(table: nx.DiGraph, expression: str) -> float:
    """Evaluate the expression and return the result
    Args:
        table: graph representing the spreedtable
        expression: reverse polish notation expression
    Returns: evaluation result float number or np.NaN when error.
    """
    logger.debug("expression: [%s]", expression)

    # An expression can include references to other nodes.
    references: List[str] = list(table.nodes())

    # --------------------------------------------------------------------------------
    # Use stack to evaluate the reverse hungarian expression
    # The pointer is the index to the current stack top which has a value.
    # If the stack is empty then point is -1.
    # --------------------------------------------------------------------------------
    stack: List[float] = []
    
    # --------------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------------
    tokens = expression.split()
    for token in tokens:
        token_type, value = evaluate_token(token=token, references=references)

        # --------------------------------------------------------------------------------
        # Terminal which cannot be further evaluated e.g. -1, 3, 5
        # --------------------------------------------------------------------------------
        if token_type == TOKEN_TYPE_TERMINAL:
            stack.append(value)
            logger.debug(
                "evaluate_expression(): token is terminal [%s]. Stack is at [%s].", 
                value, len(stack)-1
            )
            
        # --------------------------------------------------------------------------------
        # Non terminal toekn that needs to be further evaluated and reduced to a terminal.
        # --------------------------------------------------------------------------------
        if token_type == TOKEN_TYPE_REFERENCE:
            reference_value = evaluate_cell(table=table, cell_name=value)
            stack.append(reference_value)
            logger.debug(
                "evaluate_expression(): token is reference to [%s] whose value is [%s]. Stack is at [%s].", 
                value, reference_value, len(stack)-1
            )
            
        # --------------------------------------------------------------------------------
        # Operations 
        # --------------------------------------------------------------------------------
        if token_type == TOKEN_TYPE_OPERATOR:
            func = value
            assert callable(func), \
                f"token type is operator. expected callable but {type(func)}"
            if not len(stack) > 1:
                logger.error(
                    "evaluate_expression(): operation requires two operands but stack size is [%s]",
                    len(stack)
                )
                # Empty the stack if there is enough arguments for the function
                if len(stack) == 1:
                    stack.pop()
                result = np.NaN
            else:
                # right operand is at the top of the stack
                # e.g. "2 3 -" => subtract(2, 3)
                right = stack.pop()
                left = stack.pop()
                result = func(left, right)

            if not np.isfinite(result):
                logger.error("evaluate_expression(): operator result is invalid: [%s]", result)

            stack.append(result)
        
        if token_type == TOKEN_TYPE_INVAID:
            logger.error("evaluate_expression(): token [%s] is invalid", token)
            del stack
            return np.NaN
        
    # --------------------------------------------------------------------------------        
    # End condition: Stack has only one value left
    # --------------------------------------------------------------------------------
    evaluated: float = np.NaN
    if len(stack) == 1:
        evaluated = stack.pop()
    else:
        logger.error(
            "evaluate_expression(): evaluation of [%s] failed with stack size [%s].", 
            expression, len(stack)
        )
        evaluated = np.NaN

    assert isinstance(evaluated, float), \
        f"expected the evaluation result of expression [{evaluated}] as float but [{type(evaluated)}]."

    return evaluated
        

def evaluate_cell(table, cell_name) -> float:
    """Evaluate the expression of the cell to reduce to a number.
    Each cell is a node of the graph representing the table

    Args:
        table: graph representing the spred table
        cell_name: name of the cell to evaluate
    Returns: evaluated cell value which has been reduced to a number
    """
    attributes: Dict[str, Any] = table.nodes()[cell_name]
    assert "expression" in attributes, "'expression' is not in the node attribute [%s]" % attributes

    expression: str = attributes['expression']
    logger.debug(
        "evaluate_cell(): cell [%s] expression [%s].", cell_name, expression
    )

    value: float = evaluate_expression(table=table, expression=expression)
    if not np.isfinite(value):
        logger.error("evaluate_cell(): cell evaluation [%s] is invalid", value)

    table.nodes[cell_name]['value'] = value
    assert isinstance(value, float), f"expected the cell value [{value}] as float but [{type(value)}]."
    return value


def process_path(table: nx.DiGraph, path: List[str]):
    """Iterate through the sequentially ordered path of cells to evaluate each cell
    Args:
        table: graph representing the table structure
        path: a reverse-topologically-sorted list of cells to evaluate in order
    """
    logger.debug("process_path(): path %s", path)
    for cell_name in path:
        value: float = evaluate_cell(table, cell_name)
        logger.debug("process_path(): cell [%s] value [%s].", cell_name, value)


def process_table(table: nx.DiGraph):
    """Process the table
    A table is a graph which consists of multiple independent sub graphs.
    Get a reverse topologically sorted list of cells from each sub graph, and
    Go through each path to evaluate cells in the path.

    Args:
        table: graph
    Returns:
        table as the updated graph. The value attribute of each node is updated with
        the cell evaluation result reduced to a number or np.NaN.
    """
    paths: List[List[str]] = list_reverse_topological_sorted_paths_in_graph(table)
    logger.debug("evaluate_table(): independent paths in the graph %s", paths)
    
    for path in paths:
        process_path(table=table, path=path)

    return table


def save_result_table(
    table: nx.DiGraph, 
    column_names: List[str], 
    row_indices: List[str],
    path_to_file: str
):
    """Output the processed table to the output CSV
    Args:
        table: graph representing the shred table
        column_names: column names of the source CSV file
        row_indices: row indicds of the source CSV file
        path_to_file: path to the CSV
    """
    with open(path_to_file, "w", encoding="utf-8") as output:
        output.write(CSV_FIELD_DELIMITER.join(column_names) + OS_EOL_CHARACTER)
        for row in row_indices:
            line: List[str] = []
            for column in column_names:
                cell_name: str = f"{column}{row}"
                value: float = table.nodes[cell_name]['value']
                logger.debug("save_result_table(): cell[%s] value[%s]", cell_name, value)
                assert isinstance(value, float), \
                    f"Node[{cell_name}]:{table.nodes()[cell_name]} has invalid value [{value}] of type [{type(value)}]"
                if np.isfinite(value):
                    line.append(str(value))
                else:
                    line.append("#ERR")

            logger.debug("save_result_table(): line to output %s", line)
            line_in_file = CSV_FIELD_DELIMITER.join(line) + OS_EOL_CHARACTER
            output.write(line_in_file)
