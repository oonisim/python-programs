"""Module for AWS Lambda operations using Boto3
Note: a.b.lambda package causes syntax error as 'lambda' cannot be used as the package name.
"""
import json
import logging
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Iterable,
)

import botocore


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger()


class LambdaFunction:
    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(self, lambda_client):
        self.lambda_client = lambda_client

    def invoke(self, function_name: str, payload: Dict, get_log: bool = False):
        """Invokes a Lambda function.
        TODO: Implement lambda alias/version enforcement, not $LATEST

        Args:
            function_name: The name of the function to invoke.
            payload:
                The parameters of the function as a dict.
                This dict is serialized to JSON before it is sent to Lambda.
            get_log:
                When true, the last 4 KB of the execution log are included in the response.
        Returns: The response from the function invocation.
        """
        try:
            response: Dict[str, Any] = self.lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps(payload, default=str, ensure_ascii=True),
                LogType='Tail' if get_log else 'None'
            )

        except botocore.exceptions.ClientError as error:
            _logger.error("Couldn't invoke function %s.", function_name)
            raise

        return response

    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
    @staticmethod
    def get_json_payload_from_event(
            event: dict,
            expect_payload_as_dictionary: bool = True,
            expected_dictionary_element_names: Iterable = None
    ) -> Union[List, Dict]:
        """Get JSON payload from event['body'].
        Prerequisite:
            Event is a dictionary with the structure:
            {
                "body": "<serialized string from json.dumps()>"
            }
            where the 'body' element is expected to be ASCII string serialized from JSON
            e.g. json.dumps(obj, ensure_ascii=True, default=str) to eliminate the encoding
            confusion and to make sure network safe.

            If the body element is assured to be an ASCII string, we can consistently use
            json.loads(event['body']) to de-serialize it to Python dictionary safely with
            no ambiguity of encoding or data corruption during the network transfer.

        Args:
            event: Event object passed as Lambda handler argument.
            expect_payload_as_dictionary:
                if expect the event['body'] as a dictionary mapping, not an array.
            expected_dictionary_element_names:
                when expect_payload_as_dictionary is True, list of names expected
                in the event['body']. Raise ValueError is expected elements are missing.

        Returns: List or Dictionary
        Raises:
            TypeError:  if event is not a dictionary.
            ValueError: if event['body'] does not exist or there is missing expected elements.
        """
        name: str = "get_payload_elements_from_event()"
        # --------------------------------------------------------------------------------
        # Validate event as dictionary
        # --------------------------------------------------------------------------------
        if not event or not isinstance(event, dict):
            msg: str = "invalid request event."
            _logger.error("%s: %s event:[%s]", name, msg, event)
            raise TypeError(msg)

        # --------------------------------------------------------------------------------
        # Validate event['body'] as serialized string from JSON.
        # --------------------------------------------------------------------------------
        if 'body' not in event:
            msg: str = "request has no payload data."
            _logger.error(
                "%s: expected 'body' element in the event argument. event\n%s",
                name,
                event
            )
            raise ValueError(msg)

        try:
            payload: Union[List, Dict] = json.loads(event['body'])
        except (json.JSONDecodeError, TypeError) as error:
            msg: str = f"request payload needs to be a string serialized from JSON, " \
                       f"got [{event['body']}] of type {type(event['body'])}."
            _logger.error(
                "%s: expected event['body'] as a string serialized from JSON, "
                "got [%s] of type [%s]. event:\n%s",
                name, event['body'], type(event['body']), event
            )
            raise ValueError(msg) from error

        # --------------------------------------------------------------------------------
        # Dictionary structure requirement for the JSON
        # --------------------------------------------------------------------------------
        if expect_payload_as_dictionary:
            if not isinstance(payload, dict):
                msg: str = f"Dictionary format is expected as the request payload, got {payload}."
                _logger.error(msg)
                raise ValueError(msg)

            if expected_dictionary_element_names is not None:
                assert isinstance(expected_dictionary_element_names, Iterable), \
                    f"expected expected_dictionary_element_names as iterable, " \
                    f"got {expected_dictionary_element_names}"

                for element_name in expected_dictionary_element_names:
                    assert isinstance(element_name, str), \
                        f"expected element names as str, got [{type(element_name)}]"

                    if element_name not in payload:
                        msg: str = f"expected '{element_name}' in the payload:\n{payload}"
                        raise ValueError(msg)

        return payload
