from app.extensions import db
from app.extensions.api import (Namespace,
                                Resource)
from .models import InvoiceModel
from .parameters import (CreateInvoiceParameters,
                         UpdateInvoiceParameters)
from .schemas import (InvoiceSchema)

ns = Namespace('invoices',
               ordered=True,
               description="Namespace for the 'Invoice' resource.")


@ns.route('')
class InvoiceResource(Resource):
    @ns.parameters(CreateInvoiceParameters())
    @ns.response(InvoiceSchema())
    def post(self, args):
        """
        Creates a Invoice
        """
        invoice = InvoiceModel(**args)
        db.session.add(invoice)
        return invoice


@ns.route('/<string:invoice_id>')
@ns.resolve_object_by_model(InvoiceModel, 'invoice')
class InvoiceByIDResource(Resource):
    @ns.response(InvoiceSchema())
    def get(self, invoice):
        """
        Get invoice details by Id.
        """
        return invoice

    @ns.parameters(UpdateInvoiceParameters())
    @ns.response(InvoiceSchema())
    def put(self, args, invoice):
        """
        Put invoice details by Id.
        """
        print(args)
        invoice.update(args)
        return invoice

    def delete(self, invoice):
        """
        Delete a invoice by Id.
        """
        db.session.delete(invoice)
        return None, 204
