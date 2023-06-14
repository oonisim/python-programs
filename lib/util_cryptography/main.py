"""Module for cryptography utility"""
import base64
from typing import (
    Tuple,
    Optional
)

from cryptography.hazmat.primitives import (
    serialization,
    hashes
)
from cryptography.hazmat.primitives.asymmetric import (
    rsa,
    padding
)
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey
)

from util_python import (
    is_base64_encoded
)


# ----------------------------------------------------------------------
# RSA cryptography
# ----------------------------------------------------------------------
class RSA:
    """Class for RSA cryptography"""
    def __init__(self):
        self._initialized: bool = True

    @staticmethod
    def generate_keys() -> Tuple[RSAPrivateKey, RSAPublicKey]:
        """Generate RSA private and public keys
        Returns: (private_key, public_key)
        """
        private_key: RSAPrivateKey = rsa.generate_private_key(
            public_exponent=65537,    # Either 65537 or 3 (for legacy purposes).
            key_size=2048
        )
        public_key: RSAPublicKey = private_key.public_key()

        return private_key, public_key

    @staticmethod
    def serialize_private_key(
        private_key: RSAPrivateKey,
        passphrase: Optional[str] = None
    ) -> bytes:
        """Serialize public key and base 64 encode it.
        Args:
            private_key: private key to seriaize
            passphrase: pass phrase to encrypt the key
        Returns: serialized private key
        """
        assert isinstance(private_key, RSAPrivateKey)
        assert isinstance(passphrase, str) or passphrase is None

        if passphrase and len(passphrase) > 0:
            encryption_algorithm = serialization.BestAvailableEncryption(
                passphrase.strip().encode()    # must be bytes
            )
        else:
            encryption_algorithm = serialization.NoEncryption()

        private_pem: bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
        private_pem_base64: bytes = base64.b64encode(private_pem)
        return private_pem_base64

    @staticmethod
    def deserialize_private_key(
        serialized_base64_pem: bytes,
        passphrase: Optional[str] = None
    ) -> RSAPrivateKey:
        """Deserialize private key
        Args:
            serialized_base64_pem: base 64 encoded serialized private key PEM.
            passphrase: private key pass phrase
        Return: private key
        """
        assert is_base64_encoded(serialized_base64_pem), "not base64 encoded"

        password: Optional[bytes] = None
        if isinstance(passphrase, str) and len(passphrase.strip()) > 0:
            password = passphrase.strip().encode()

        serialized_pem: bytes = base64.b64decode(serialized_base64_pem)
        private_key = serialization.load_pem_private_key(
            serialized_pem,
            password=password,    # must be bytes
        )
        return private_key

    @staticmethod
    def serialize_public_key(public_key: RSAPublicKey) -> bytes:
        """Serialize publie key
        Args:
            public_key: public key to seriaize
        Returns: serialized public key
        """
        assert isinstance(public_key, RSAPublicKey)
        public_pem: bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        public_pem_base64: bytes = base64.b64encode(public_pem)
        return public_pem_base64

    @staticmethod
    def deserialize_public_key(
        serialized_base64_pem: bytes
    ) -> RSAPublicKey:
        """restore private key from the serialized and base654 encoded bytes
        Args:
            serialized_base64_pem: public key that has been serialized and base64 encodced
        Returns: public key
        """
        assert is_base64_encoded(serialized_base64_pem), "not base64 encoded"
        public_key_pem = base64.b64decode(serialized_base64_pem)
        public_key: RSAPublicKey = serialization.load_pem_public_key(
            public_key_pem

        )
        return public_key

    @staticmethod
    def save_keys(
        private_key: RSAPrivateKey,

        public_key: RSAPublicKey,
        passphrase: Optional[str] = None,
        path_to_private_key_pem_file: str = "private_rsa_key.pem",
        path_to_public_key_pem_file: str = "public_rsa_key.pem",
    ) -> Tuple[str, str]:
        """Serialize and save keys to PEM files.
        Args:
            private_key: RSA private key
            public_key: RSA public key
            passphrase: optionla private key pass phrase
            path_to_private_key_pem_file: PEM file path to save the private key
            path_to_public_key_pem_file: PEM file path to save the public key
        Returns: (path_to_private_key_pem_file, path_to_public_key_pem_file)
        Raises:

            RuntimeError: Cannot save to the PEM files
        """
        # private key
        private_pem: bytes = RSA.serialize_private_key(
            private_key=private_key, passphrase=passphrase
        )
        # public key
        public_pem: bytes = RSA.serialize_public_key(public_key=public_key)

        try:
            with open(path_to_private_key_pem_file, 'wb') as pem:

                pem.write(private_pem)
            with open(path_to_public_key_pem_file, 'wb') as pem:
                pem.write(public_pem)

        except OSError as error:
            msg: str = f"key serialization failed due to {str(error)}."
            raise RuntimeError(msg) from error

        return path_to_private_key_pem_file, path_to_public_key_pem_file

    @staticmethod
    def load_private_key(
        path_to_private_key_pem_file: str = "private_rsa_key.pem",
        passphrase: Optional[str] = None
    ) -> RSAPrivateKey:
        """Load RSA private key from a PEM file
        Args:
            path_to_private_key_pem_file: PEM file path to load the private key
            passphrase: password to encrypt the private key
        Returns: RSAPrivateKey
        Raises:

            RuntimeError: Cannot load from the PEM file.
        """
        try:
            with open(file=path_to_private_key_pem_file, mode="rb") as key_file:
                private_key: RSAPrivateKey = RSA.deserialize_private_key(
                    serialized_base64_pem=key_file.read(),
                    passphrase=passphrase
                )
                return private_key

        except OSError as error:
            msg: str = f"failed to load from [{path_to_private_key_pem_file}] "\
                       f"due to [{error}]."
            raise RuntimeError(msg) from error

    @staticmethod
    def load_public_key(
        path_to_public_key_pem_file: str = "public_rsa_key.pem",
    ) -> RSAPublicKey:
        """Load RSA public key from a PEM file
        Args:
            path_to_public_key_pem_file: PEM file path to load the public key
        Returns: RSAPublicKey
        Raises:

            RuntimeError: Cannot load from the PEM file.
        """
        try:
            with open(file=path_to_public_key_pem_file, mode="rb") as key_file:
                public_key = RSA.deserialize_public_key(
                    serialized_base64_pem=key_file.read()
                )
            return public_key

        except OSError as error:
            msg: str = f"failed to load from [{path_to_public_key_pem_file}] "\
                       f"due to [{error}]."
            raise RuntimeError(msg) from error

    @staticmethod
    def encrypt(
        data: bytes,
        public_key: RSAPublicKey
    ) -> bytes:
        """Encrypt the message with the public key and BASE64 encode it to be network safe.
        Args:
            data: data of bytes type to encrypt
            public_key: key to use
        Returns: base64 encoded encypted bytes
        """
        assert isinstance(data, bytes) and len(data) > 0
        encrypted = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        encrypted_base64: bytes = base64.b64encode(encrypted)
        return encrypted_base64

    @staticmethod
    def decrypt(
        encrypted_base64: bytes,
        private_key: RSAPrivateKey
    ) -> bytes:
        """Decrypt the base64 encoded encrypted message with the private key.
        Args:
            encrypted_base64: base 64 encoded encrypted data
            private_key: key to use
        Returns: decrypted string, not bytes
        """
        assert isinstance(encrypted_base64, bytes) and len(encrypted_base64) > 0
        assert is_base64_encoded(encrypted_base64), "encrypted is not base64 encoded."
        decrypted = private_key.decrypt(
            base64.b64decode(encrypted_base64),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted
