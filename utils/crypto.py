import os
import json
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Descifrado compatible con el crypto de dumbo-prod (Web Crypto API, AES-256-GCM).
# El payload cifrado tiene el formato base64 "IV:EncryptedData:AuthTag", igual que
# produce serverEncrypt/encrypt en el front (src/utils/crypto.server.ts y crypto.v3.ts).
#
# IMPORTANTE sobre la clave: el TS hace `baseKey.padEnd(32,'0').slice(0,32)` y luego
# `TextEncoder().encode(...)`. Es decir, toma los primeros 32 CARACTERES del string
# de la clave y los codifica como UTF-8 (NO decodifica base64). Aquí replicamos ese
# comportamiento exacto para que las claves coincidan byte a byte.


def _get_key() -> bytes:
    base_key = os.getenv("ENCRYPTION_KEY")
    if not base_key:
        raise RuntimeError("ENCRYPTION_KEY is not set in the environment variables")
    key_string = base_key.ljust(32, "0")[:32]
    return key_string.encode("utf-8")


def decrypt(encrypted_data: str) -> str:
    """Descifra un string con formato base64 'IV:EncryptedData:AuthTag' (AES-256-GCM)."""
    parts = encrypted_data.split(":")
    if len(parts) != 3:
        raise ValueError("Invalid encrypted data format")

    iv = base64.b64decode(parts[0])
    encrypted = base64.b64decode(parts[1])
    auth_tag = base64.b64decode(parts[2])

    # AESGCM.decrypt espera ciphertext + tag concatenados (así los agrupa Web Crypto).
    combined = encrypted + auth_tag
    aesgcm = AESGCM(_get_key())
    decrypted = aesgcm.decrypt(iv, combined, None)
    return decrypted.decode("utf-8")


def decrypt_body(encrypted_data: str) -> dict:
    """Descifra y parsea como JSON el objeto original."""
    return json.loads(decrypt(encrypted_data))
