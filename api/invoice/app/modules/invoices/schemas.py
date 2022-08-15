import marshmallow.fields as ma

from app.extensions.api import ModelSchema


class ContactSchema(ModelSchema):
    contact_id = ma.Str(dump_to='ContactID')
    contact_number = ma.Str(dump_to='ContactNumber')
    contact_status = ma.Str(dump_to='ContactStatus')
    name = ma.Str(dump_to='Name')


class LineItemSchema(ModelSchema):
    description = ma.Str(dump_to='Description')
    quantity = ma.Number(dump_to='Quantity')
    unit_amount = ma.Number(dump_to='UnitAmount')


class PaymentSchema(ModelSchema):
    date = ma.Date(dump_to='Date')
    amount = ma.Number(dump_to='Amount')
    payment_id = ma.Number(dump_to='PaymentID')


class CreditNoteSchema(ModelSchema):
    type = ma.Str(dump_to='Type')
    contact = ma.Nested(ContactSchema, dump_to='Contact')
    date = ma.Date(dump_to='Date')
    status = ma.String(dump_to='Status')
    credit_note_id = ma.Number(dump_to='CreditNoteID')


class PrepaymentSchema(ModelSchema):
    type = ma.Str(dump_to='Type')
    contact = ma.Nested(ContactSchema, dump_to='Contact')
    date = ma.Date(dump_to='Date')
    status = ma.String(dump_to='Status')
    prepayment_id = ma.Number(dump_to='PrepaymentID')


class OverpaymentSchema(ModelSchema):
    type = ma.Str(dump_to='Type')
    contact = ma.Nested(ContactSchema, dump_to='Contact')
    date = ma.Date(dump_to='Date')
    status = ma.String(dump_to='Status')
    overpayment_id = ma.Number(dump_to='OverpaymentID')


class InvoiceSchema(ModelSchema):
    invoice_id = ma.Str(dump_to='InvoiceID')
    type = ma.Str(dump_to='Type')
    contact = ma.Nested(ContactSchema, dump_to='Contact')
    date = ma.Date(dump_to='Date')
    due_date = ma.Date(dump_to='DueDate')
    status = ma.Str(dump_to='Status')
    line_amount_types = ma.Str(dump_to='LineAmountTypes')
    line_items = ma.List(ma.Nested(LineItemSchema), dump_to='LineItems')
    sub_total = ma.Number(dump_to='SubTotal')
    total_tax = ma.Number(dump_to='TotalTax')
    total = ma.Number(dump_to='Total')
    total_discount = ma.Number(dump_to='Total_Discount')
    updated_date_utc = ma.Date(dump_to='UpdatedDateUtc')
    currency_code = ma.Str(dump_to='CurrencyCode')
    currency_rate = ma.Number(dump_to='CurrencyRate')
    invoice_number = ma.Str(dump_to='InvoiceNumber')
    reference = ma.Str(dump_to='Reference')
    branding_theme_id = ma.Str(dump_to='BrandingThemeId')
    url = ma.Str(dump_to='Url')
    sent_to_contact = ma.Boolean(dump_to='SentToContact')
    expected_payment_date = ma.Date(dump_to='ExpectedPaymentDate')
    planned_payment_date = ma.Date(dump_to='PlannedPaymentDate')
    has_attachments = ma.Boolean(dump_to='HasAttachments')
    payments = ma.List(ma.Nested(PaymentSchema), dump_to='Payments')
    credit_notes = ma.List(ma.Nested(CreditNoteSchema), dump_to='CreditNotes')
    prepayments = ma.List(ma.Nested(PrepaymentSchema), dump_to='Prepayments')
    overpayments = ma.List(ma.Nested(OverpaymentSchema), dump_to='Overpayments')
    amount_due = ma.Number(dump_to='AmountDue')
    amount_paid = ma.Number(dump_to='AmountPaid')
    cis_deduction = ma.Number(dump_to='CisDeduction')
    fully_paid_on_date = ma.Date(dump_to='FullyPaidOnDate')
    amount_credited = ma.Number(dump_to='AmountCredited')
