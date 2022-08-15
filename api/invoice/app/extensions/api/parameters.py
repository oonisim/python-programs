"""
app/extensions/api/parameters.py
Overrides Flask-Restplus Schema class to Paramters class
Forked from https://github.com/frol/flask-restplus-server-example flask-restplus-patched folder
"""
from flask_marshmallow import Schema, base_fields
from marshmallow import (validate,
                         validates_schema,
                         ValidationError)


class Parameters(Schema):
    class Meta:
        ordered = True

    def __init__(self, **kwargs):
        super(Parameters, self).__init__(strict=True, **kwargs)
        # This is an add-hoc implementation of the feature which didn't make
        # into Marshmallow upstream:
        # https://github.com/marshmallow-code/marshmallow/issues/344
        for required_field_name in getattr(self.Meta, 'required', []):
            self.fields[required_field_name].required = True

    def __contains__(self, field):
        return field in self.fields

    def make_instance(self, data):
        # pylint: disable=unused-argument
        """
        This is a no-op function which shadows ``ModelSchema.make_instance``
        method (when inherited classes inherit from ``ModelSchema``). Thus, we
        avoid a new instance creation because it is undesirable behaviour for
        parameters (they can be used not only for saving new instances).
        """
        return


class PaginationParameters(Parameters):
    """
    Helper Parameters class to reuse pagination.
    """
    per_page = base_fields.Integer(
        description="per_page a number of items (allowed range is 1-100), default is 20.",
        missing=20,
        validate=validate.Range(min=1, max=100)
    )
    page = base_fields.Integer(
        description="page number, default is 1.",
        missing=1,
        validate=validate.Range(min=1)
    )


class SearchParameters(PaginationParameters):
    """
    Helper Parameters class to reuse search.
    """
    sort_by = base_fields.String(
        description="Fields to be sorted.",
        required=False,
        location='querystring',
        validate=validate.OneOf([]))
    sort_direction = base_fields.String(
        description="Type of the sorting, 'asc' or 'desc'.",
        required=False,
        location='querystring',
        validate=validate.OneOf(['asc', 'desc']))

    @validates_schema
    def validate_sorting_params(self, data):
        if bool(data.get('sort_by')) != bool(data.get('sort_direction')):
            raise ValidationError("Payload must contains both 'sort_by' and 'sort_direction' if one or the other is present.")
