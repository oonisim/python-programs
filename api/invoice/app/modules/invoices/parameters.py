import marshmallow.fields as ma

from app.extensions.api import Parameters


class LineItemParameters(Parameters):
    Description = ma.Str(required=True, location='json', attribute='description')
    Quantity = ma.Number(required=True, location='json', attribute='quantity')
    UnitAmount = ma.Number(required=True, location='json', attribute='unit_amount')


class CreateInvoiceParameters(Parameters):
    Type = ma.String(required=True, location='json', attribute='type')
    ContactId = ma.String(required=True, location='json', attribute='contact_id')
    Status = ma.String(required=False, location='json', attribute='status')
    LineItems = ma.List(ma.Nested(LineItemParameters), required=True, location='json', attribute='line_items')


class UpdateInvoiceParameters(Parameters):
    """
    Parameters to update Invoice
    """
    Type = ma.String(required=False, location='json', attribute='type')
    ContactId = ma.String(required=False, location='json', attribute='contact_id')
    Status = ma.String(required=False, location='json', attribute='status')
    LineItems = ma.List(ma.Nested(LineItemParameters), required=True, location='json', attribute='line_items')
