"""Module to test KMS class"""
import os
from typing import (
    Dict,
    Any
)

from util_aws.boto3.kms import (
    KMSRSAEncryptDecrypt
)

# --------------------------------------------------------------------------------
# AWS
# Prerequisite:
# KMS RSA key for encryption/decryption has been created and available in the region.
# --------------------------------------------------------------------------------
AWS_KMS_KEY_ID: str = "6fcd4039-37c9-4408-b724-deffa5a9c7db"
os.environ['AWS_DEFAULT_REGION'] = "us-east-2"


# --------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------
def test_describe_key():
    """Test if the KMS KEY is RSA for encryption/decryption usage"""
    _kms: KMSRSAEncryptDecrypt = KMSRSAEncryptDecrypt(
        kms_key_id=AWS_KMS_KEY_ID
    )
    try:
        description: Dict[str, Any] = _kms.describe_key()
        assert description['KeyMetadata']['Enabled'], \
            f"KMS key [{AWS_KMS_KEY_ID}] not enabled."

        assert (
                description['KeyMetadata']['KeySpec'] in
                KMSRSAEncryptDecrypt.RSA_KEY_SPECIFICATIONS
        ), f"KMS Spec {description['KeyMetadata']['KeySpec']} " \
           f"not in {KMSRSAEncryptDecrypt.RSA_KEY_SPECIFICATIONS}"

        assert description['KeyMetadata']['KeyState'] == 'Enabled', \
            f"KMS key [{AWS_KMS_KEY_ID}] not active " \
            f"because in {[description['KeyMetadata']['KeyState']]} state."

        assert description['KeyMetadata']['KeyUsage'] == 'ENCRYPT_DECRYPT', \
            f"KMS key [{AWS_KMS_KEY_ID}] usage is not encrypt/decrypt."

        assert (
            KMSRSAEncryptDecrypt.RSA_ENCRYPTION_ALGORITHM in
            description['KeyMetadata']['EncryptionAlgorithms']
        ), "RSAES_OAEP_SHA_256 algorithm is not available, verify correct one to use."

    except RuntimeError as error:
        assert False, error


def test_encrypt_decrypt():
    """Test if decrypt can restore the original message from encrypted"""
    _kms: KMSRSAEncryptDecrypt = KMSRSAEncryptDecrypt(
        kms_key_id=AWS_KMS_KEY_ID
    )
    secret: str = "takoikabin"
    encrypted = _kms.encrypt(data=secret.encode())
    # print(encrypted.decode('utf-8'))
    decrypted = _kms.decrypt(data=encrypted)
    # print(decrypted.decode('utf-8'))
    assert decrypted.decode('utf-8') == secret, \
        f"expected [{secret}], got [{decrypted.decode('utf-8')}]."
