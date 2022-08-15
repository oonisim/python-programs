from app.extensions import db
import uuid


class InvoiceModel(db.Model):
    __tablename__ = 'Invoice'
    invoice_id = db.Column('InvoiceId', db.String(length=36), primary_key=True)
    type = db.Column('Type', db.String(length=9), nullable=False)
    contact_id = db.Column('ContactId', db.String(length=36), db.ForeignKey('Contact.ContactId'), nullable=False)
    date = db.Column('Date', db.Date)
    due_date = db.Column('DueDate', db.Date)
    status = db.Column('Status', db.String(length=10))
    line_amount_types = db.Column('LineAmountTypes', db.String(length=10))
    line_items = db.relationship("LineItemModel", back_populates='invoice', cascade='all')
    sub_total = db.Column('SubTotal', db.Numeric(precision=20, scale=2))
    total_tax = db.Column('TotalTax', db.Numeric(precision=20, scale=2))
    total = db.Column('Total', db.Numeric(precision=20, scale=2))
    total_discount = db.Column('Total_Discount', db.Numeric(precision=20, scale=2))
    updated_date_utc = db.Column('UpdatedDateUtc', db.Date)
    currency_code = db.Column('CurrencyCode', db.String(length=3))
    currency_rate = db.Column('CurrencyRate', db.Numeric(precision=20, scale=2))
    invoice_number = db.Column('InvoiceNumber', db.String(length=36))
    reference = db.Column('Reference', db.String(length=100))
    branding_theme_id = db.Column('BrandingThemeId', db.String(length=36),
                                  db.ForeignKey('BrandingTheme.BrandingThemeId'))
    url = db.Column('Url', db.String(length=60))
    sent_to_contact = db.Column('SentToContact', db.Boolean)
    expected_payment_date = db.Column('ExpectedPaymentDate', db.Date)
    planned_payment_date = db.Column('PlannedPaymentDate', db.Date)
    has_attachments = db.Column('HasAttachments', db.Boolean)
    payments = db.relationship("PaymentModel", back_populates='invoice', cascade='all')
    credit_notes = db.relationship("CreditNoteModel", back_populates='invoice', cascade='all')
    prepayments = db.relationship("PrepaymentModel", back_populates='invoice', cascade='all')
    overpayments = db.relationship("OverpaymentModel", back_populates='invoice', cascade='all')
    amount_due = db.Column('AmountDue', db.Numeric(precision=20, scale=2))
    amount_paid = db.Column('AmountPaid', db.Numeric(precision=20, scale=2))
    cis_deduction = db.Column('CisDeduction', db.Numeric(precision=20, scale=2))
    fully_paid_on_date = db.Column('FullyPaidOnDate', db.Date)
    amount_credited = db.Column('AmountCredited', db.Numeric(precision=20, scale=2))

    def __init__(self, **kwargs):
        super(InvoiceModel, self).__init__(**kwargs)
        self.invoice_id = str(uuid.uuid4()).upper()
        if self.status is None:
            self.status = 'DRAFT'

    def update(self, args):
        self.checkIfCanUpdate(args)

        if args.get('status') == 'PAID':
            raise Exception("can't change paid invoice")

        if len(self.credit_notes) > 0:
            raise Exception("part-payments or credit notes applied")

        if len(self.payments) > 0:
            raise Exception("part-payments or credit notes applied")

    def checkIfCanUpdate(self, args):
        self.type = args.get('type', self.type)

        if args.get('status') != self.status:
            if self.status == 'DRAFT':
                if args.get('status') not in ('SUBMITTED', 'AUTHORISED', 'DELETED'):
                    raise Exception("bad")
            elif self.status == 'SUBMITTED':
                if args.get('status') not in ('SUBMITTED', 'AUTHORISED', 'DRAFT', 'DELETED'):
                    raise Exception("bad")
            elif self.status == 'AUTHORIZED':
                if args.get('status') not in ('VOID'):
                    raise Exception("bad")

        self.status = args.get('status', self.status)

        self.contact_id = args.get('contact_id')
        self.line_items = args.get('line_items')


class ContactModel(db.Model):
    __tablename__ = 'Contact'
    contact_id = db.Column('ContactId', db.String(length=36), primary_key=True)


class BrandingThemeModel(db.Model):
    __tablename__ = 'BrandingTheme'
    contact_id = db.Column('BrandingThemeId', db.String(length=36), primary_key=True)


class LineItemModel(db.Model):
    __tablename__ = 'LineItem'
    line_item_id = db.Column('LineItemId', db.String(length=36), primary_key=True)
    invoice_id = db.Column('InvoiceId', db.String(length=36), db.ForeignKey('Invoice.InvoiceId'), nullable=False)
    invoice = db.relationship("InvoiceModel", back_populates='line_items', cascade='all')
    description = db.Column('Description', db.String(length=50))
    quantity = db.Column('Quantity', db.Numeric(precision=20, scale=2))
    unit_amount = db.Column('UnitAmount', db.Numeric(precision=20, scale=2))


class PaymentModel(db.Model):
    __tablename__ = 'Payment'
    payment_id = db.Column('PaymentId', db.String(length=36), primary_key=True)
    invoice_id = db.Column('InvoiceId', db.String(length=36), db.ForeignKey('Invoice.InvoiceId'), nullable=False)
    invoice = db.relationship("InvoiceModel", back_populates='payments', cascade='all')


class CreditNoteModel(db.Model):
    __tablename__ = 'CreditNote'
    credit_note_id = db.Column('CreditNoteId', db.String(length=36), primary_key=True)
    invoice_id = db.Column('InvoiceId', db.String(length=36), db.ForeignKey('Invoice.InvoiceId'), nullable=False)
    invoice = db.relationship("InvoiceModel", back_populates='credit_notes', cascade='all')


class PrepaymentModel(db.Model):
    __tablename__ = 'Prepayment'
    prepayment_id = db.Column('PrepaymentId', db.String(length=36), primary_key=True)
    invoice_id = db.Column('InvoiceId', db.String(length=36), db.ForeignKey('Invoice.InvoiceId'), nullable=False)
    invoice = db.relationship("InvoiceModel", back_populates='prepayments', cascade='all')


class OverpaymentModel(db.Model):
    __tablename__ = 'Overpayment'
    overpayment_id = db.Column('OverpaymentId', db.String(length=36), primary_key=True)
    invoice_id = db.Column('InvoiceId', db.String(length=36), db.ForeignKey('Invoice.InvoiceId'), nullable=False)
    invoice = db.relationship("InvoiceModel", back_populates='overpayments', cascade='all')
