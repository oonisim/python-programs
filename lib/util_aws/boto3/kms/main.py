"""Module for AWS KMS operations
https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html

[KMS]
NOTE: KMS RSA key usage is exclusive between
1. Encrypt and decrypt
2. Sign and Verify

Although RSA can do both, KMS does not allow KMS RSA key to do both.
For encryption, decryption, need to specify the algorithm. For 2048 length, either:
1. RSAES_OAEP_SHA_1
2. RSAES_OAEP_SHA_256

[RSA]
NOTE: Data length limit.
RSA is only able to encrypt data to a maximum amount equal to your key size
(2048 bits = 256 bytes), minus padding and header data (11 bytes for PKCS#1 v1.5 padding).
If the data length exceeds the limit, KMS throws "cannot encrypt/decrypt" error
without no indication of length limit.
"""
import base64
import json
import logging
from typing import (
    List,
    Dict,
    Set,
    Any,
    Optional
)

from util_aws.boto3.common import (     # pylint: disable=import-error
    Base
)
from util_logging import (              # pylint: disable=import-error
    get_logger
)
from util_python import (               # pylint: disable=import-error
    is_base64_encoded
)

import boto3


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# AWS KMS
# --------------------------------------------------------------------------------
class KMSRSAEncryptDecrypt(Base):
    """Class to provide AWS KMS RSA asymmetric key encryption/decryption functions.
    Note that for encryption/decryption, the KMS key usage needs to have 
    been set to Encryption/Decryption.
    """
    RSA_ENCRYPTION_ALGORITHM: str = 'RSAES_OAEP_SHA_256'
    RSA_KEY_SPECIFICATIONS: Set[str] = {'RSA_2048', 'RSA_3072', 'RSA_4096'}

    # pylint: disable=too-few-public-methods
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
    @staticmethod
    def create_key(
            keyspec: str,
            description: Optional[str] = None,
            tags: Optional[List[Dict]]= None
    ) -> Dict[str, Any]:
        """Create KMS RSA key
        Args:
            keyspec: 'RSA_2048'|'RSA_3072'|'RSA_4096'
            description: explanation of the key
            tags: AWS resource tag in the format: [
                {
                    'TagKey': 'string',
                    'TagValue': 'string'
                },
                ...
            ]
        Returns: {
            "KeyId": KeyId,
            "Arn": Arn,
            "EncryptionAlgorithms": [EncryptionAlgorithms],
            "CreationDate": CreationDate datetime instance,
            "ValidTo": ValidTo datetime instance
        }
        """
        _func_name: str = "create_key()"
        assert tags is None or isinstance(tags, List) and isinstance(tags[0], Dict)
        if keyspec.upper() not in KMSRSAEncryptDecrypt.RSA_KEY_SPECIFICATIONS:
            msg: str = f"expected one of {KMSRSAEncryptDecrypt.RSA_KEY_SPECIFICATIONS}, " \
                  f"got {keyspec}."
            _logger.error("%s: %s", _func_name, msg)
            raise ValueError(msg)

        _client = boto3.client('kms')
        try:
            response = _client.create_key(
                KeySpec=keyspec,
                KeyUsage='ENCRYPT_DECRYPT',
                Description=description,
                Tags=tags
            )
            assert response['KeyMetadata']['KeySpec'] == keyspec.upper(), \
                f"invalid KeySpec {response['KeyMetadata']['KeySpec']}"
            assert response['KeyMetadata']['KeyUsage'] == 'ENCRYPT_DECRYPT', \
                f"invalid KeyUsage {response['KeyMetadata']['KeyUsage']}"
            assert response['KeyMetadata']['Enabled'], \
                f"invalid status {response['KeyMetadata']['Enabled']}"

            return {
                "KeyId": response['KeyMetadata']['KeyId'],
                "Arn": response['KeyMetadata']['Arn'],
                "EncryptionAlgorithms": response['KeyMetadata']['EncryptionAlgorithms'],
                "CreationDate": response['KeyMetadata']['CreationDate'],
                "ValidTo": response['KeyMetadata']['ValidTo']
            }

        except _client.exceptions.TagException as error:
            msg: str = f"KMS key creation failed due to {error} with tags:{tags}."
            _logger.error("%s: %s", _func_name, msg)
            raise ValueError(msg) from error
        except (
            _client.exceptions.LimitExceededException,
            _client.exceptions.KMSInternalException,
            _client.exceptions.DependencyTimeoutException
        ) as error:
            msg: str = f"KMS key creation failed due to {error}."
            _logger.error("%s: %s", _func_name, msg)
            raise RuntimeError(msg) from error

        finally:
            del _client

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    @property
    def encryption_algorithm(self) -> str:
        """EncryptionAlgorithm for the key
        AWS KMS requires EncryptionAlgorithm for asymmetric key.
        """
        assert self._encryption_algorithm, \
            f"expected encryption_algorithm, got {self._encryption_algorithm}."
        return self._encryption_algorithm

    def __init__(
        self,
        kms_key_id: str,
        kms_client=None
    ):
        """
        Args:
            kms_client: A Boto3 KMS client.
            kms_key_id: ID of the KMS key to use as default
        """
        super().__init__()
        self._client = kms_client if kms_client else boto3.client('kms')
        self._key_id: str = kms_key_id
        self._encryption_algorithm: str = KMSRSAEncryptDecrypt.RSA_ENCRYPTION_ALGORITHM

        # --------------------------------------------------------------------------------
        # Validate KMS key is for RSA encrypt/decrypt
        # --------------------------------------------------------------------------------
        msg: Optional[str] = None
        description: Dict[str, Any] = self.describe_key()
        if not description['KeyMetadata']['Enabled']:
            msg = f"KMS key [{self._key_id}] not enabled."
        if not description['KeyMetadata']['KeyState'] == 'Enabled':
            msg = f"KMS key [{self._key_id}] not active because " \
                  f"in {[description['KeyMetadata']['KeyState']]} state."
        if not description['KeyMetadata']['KeyUsage'] == 'ENCRYPT_DECRYPT':
            msg = f"KMS key [{self._key_id}] usage is not encrypt/decrypt."
        if not KMSRSAEncryptDecrypt.RSA_ENCRYPTION_ALGORITHM in description['KeyMetadata']['EncryptionAlgorithms']:
            msg = "RSAES_OAEP_SHA_256 algorithm is not available, verify correct one to use."
        if msg:
            raise RuntimeError(msg)

        _logger.info(
            "%s: KMS Key description: %s",
            KMSRSAEncryptDecrypt.__name__,
            json.dumps(description, indent=4, default=str, ensure_ascii=False)
        )

    def describe_key(self):
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kms/client/describe_key.html
        Returns: key description
        Raises:
            ValueError: Invalid KMS key ID
        """
        _func_name: str = "describe_key()"
        try:
            response = self._client.describe_key(
                KeyId=self._key_id,
                GrantTokens=[]
            )
            return response
        except (
                self._client.exceptions.NotFoundException,
                self._client.exceptions.InvalidArnException
        ) as error:
            msg: str = f"KMS key [{self._key_id}] does not exist in [{self._aws_region}]."
            _logger.error("%s: %s", _func_name, msg)
            raise ValueError(msg) from error

        except self._client.exceptions.KMSInternalExceptiona as error:
            msg: str = f"AWS KMS internal error with KMS key [{self._key_id}] in [{self._aws_region}]."
            _logger.error("%s: %s", _func_name, msg)
            raise RuntimeError(msg) from error

    def encrypt(
            self, data: bytes
    ) -> bytes:
        """Encrypt data with the KMS key and base64 encode the encrypted data to be network safe ASCII.

        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kms/client/encrypt.html
        Args:
            data: data to encrypt
        Returns: base64 encoded encrypted data
        """
        _func_name: str = "encrypt()"
        assert isinstance(data, bytes), f"expected bytes, got [{type(data)}]."
        assert len(data) > 0
        try:
            response = self._client.encrypt(
                KeyId=self._key_id,
                EncryptionAlgorithm=self.encryption_algorithm,
                Plaintext=data                               # Must be bytes, not string
            )
            encrypted: bytes = response['CiphertextBlob']
            encrypted_base64: bytes = base64.b64encode(encrypted)
            return encrypted_base64

        except (
                self._client.exceptions.NotFoundException,
                self._client.exceptions.InvalidArnException
        ) as error:
            msg: str = f"KMS key [{self._key_id}] does not exist in [{self._aws_region}]."
            _logger.error("%s: %s", _func_name, msg)
            raise ValueError(msg) from error

        except self._client.exceptions.KMSInternalExceptiona as error:
            msg: str = f"AWS KMS internal error with KMS key [{self._key_id}] in [{self._aws_region}]."
            _logger.error("%s: %s", _func_name, msg)
            raise RuntimeError(msg) from error

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt the encrypted and base64 encoded data with the KMS key
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kms/client/decrypt.html
        Args:
            data: data to encrypt
        """
        _func_name: str = "decrypt()"
        assert isinstance(data, bytes), f"expected bytes, got [{type(data)}]."
        assert len(data) > 0
        assert is_base64_encoded(data=data), "expected base64 encoded data."

        try:
            response = self._client.decrypt(
                KeyId=self._key_id,
                EncryptionAlgorithm=self.encryption_algorithm,
                CiphertextBlob=base64.b64decode(data)   # Must be bytes, not string
            )
            decrypted: bytes = response['Plaintext']
            return decrypted

        except (
                self._client.exceptions.NotFoundException,
                self._client.exceptions.InvalidArnException
        ) as error:
            msg: str = f"KMS key [{self._key_id}] does not exist in [{self._aws_region}]."
            _logger.error("%s: %s", _func_name, msg)
            raise ValueError(msg) from error

        except self._client.exceptions.KMSInternalException as error:
            msg: str = f"AWS KMS internal error with KMS key [{self._key_id}] in [{self._aws_region}]."
            _logger.error("%s: %s", _func_name, msg)
            raise RuntimeError(msg) from error
