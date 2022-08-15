import json
import pytest


@pytest.fixture()
def invoice_test_1(temp_db_instance_helper):
    from app.modules.invoices.models import InvoiceModel

    for invoice in temp_db_instance_helper(InvoiceModel(invoice_id="test1",
                                                        type="Invoice One",
                                                        contact_id="Lady Gaga",
                                                        status="DRAFT")):
        yield invoice


def test_create_invoice(flask_app_client):
    create_response = flask_app_client.post('/api/invoices', json={
        "Type": "Invoice Test",
        "ContactId": "Ada Lovelace",
        "Status": "DRAFT",
        "LineItems": [{
            "Description": "first line",
            "Quantity": 1,
            "UnitAmount": 5.6
        }]
    })

    assert create_response.status_code == 200
    created_invoice = json.loads(create_response.data.decode('utf-8'))
    assert created_invoice['Type'] == "Invoice Test"
    assert created_invoice['Status'] == "DRAFT"

    fetch_response = flask_app_client.get('/api/invoices/' + created_invoice['InvoiceId'])
    assert fetch_response.status_code == 200
    fetched_invoice = json.loads(fetch_response.data.decode('utf-8'))
    assert fetched_invoice['Type'] == "Invoice Test"
    assert fetched_invoice['Status'] == "DRAFT"


def test_update_invoice(flask_app_client, invoice_test_1):
    create_response = flask_app_client.put('/api/invoices/' + invoice_test_1.invoice_id, json={
        "Type": "Invoice Test",
        "ContactId": "Ada Lovelace",
        "Status": "SUBMITTED"
    })

    assert create_response.status_code == 200
    created_invoice = json.loads(create_response.data.decode('utf-8'))
    assert created_invoice['Type'] == "Invoice Test"
    assert created_invoice['Status'] == "DRAFT"

    fetch_response = flask_app_client.get('/api/invoices/' + invoice_test_1.invoice_id)
    assert fetch_response.status_code == 200
    fetched_invoice = json.loads(fetch_response.data.decode('utf-8'))
    assert fetched_invoice['Type'] == "Invoice Test"
    assert fetched_invoice['Status'] == "DRAFT"