"""Module for AWS KMS operations"""
from typing import (
    Optional
)
import boto3


# --------------------------------------------------------------------------------
# AWS KMS
# --------------------------------------------------------------------------------
class KMSRSA:
    """Class to provide AWS KMS RSA asymmetric key functions.
    Note that for encryption/decryption, the KMS key usage needs to have 
    been set to Encryption/Decryption.
    """
    # pylint: disable=too-few-public-methods
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(
        self,
        kms_client=None,
        default_kms_key_id: Optional[str] = None
    ):
        """
        Args:
            kms_client: A Boto3 KMS client.
            default_kms_key_id: ID of the KMS key to use as default
        """
        self._client = kms_client if kms_client else boto3.client('kms')
        self._key_id: Optional[str] = default_kms_key_id

    def encrypt(
            self, data: bytes, key_id: str = None
    ) -> bytes:
        """Encrypt data with the KMS key
        Args:
            data: data to encrypt
            key_id: KMS key. If None, use the default KMS key.
        """
        assert isinstance(data, bytes), f"expected bytes, got [{type(data)}]."
        assert len(data) > 0
        assert self._key_id or key_id, "no KMS key available"

        key_id: str = key_id if key_id else self._key_id
        response = self._client.encrypt(
            KeyId=key_id,
            EncryptionAlgorithm="RSAES_OAEP_SHA_256",    # algorithm is mandatory for Asymmetric
            Plaintext=data                               # Must be bytes, not string
        )
        encrypted = response['CiphertextBlob']
        return encrypted

    def decrypt(self, data: bytes, key_id: str = None) -> bytes:
        """Encrypt data with the KMS key
        Args:
            data: data to encrypt
            key_id: KMS key. If None, use the default KMS key.
        """
        assert isinstance(data, bytes), f"expected bytes, got [{type(data)}]."
        assert len(data) > 0
        assert self._key_id or key_id, "no KMS key available"

        response = self._client.decrypt(
            KeyId=key_id,
            EncryptionAlgorithm="RSAES_OAEP_SHA_256",   # algorithm is mandatory for Asymmetric
            CiphertextBlob=data                         # Must be bytes, not string
        )
        decrypted: bytes = response['Plaintext']
        return decrypted
