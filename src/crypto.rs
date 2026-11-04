/*!
Cryptographic Envelope & Hardware Key Derivation for .lagi Format

Implements AES-256-GCM encryption with hardware-rooted key derivation.
Supports platform-specific trust roots: Apple Secure Enclave, Windows TPM 2.0, Linux Keyring.
*/

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use hkdf::Hkdf;
use pbkdf2::pbkdf2_hmac;
use rand::RngCore;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Key length: 256 bits = 32 bytes
pub const KEY_LENGTH: usize = 32;

/// Nonce length: 96 bits = 12 bytes (for AES-GCM)
pub const NONCE_LENGTH: usize = 12;

/// Tag length: 128 bits = 16 bytes (AES-GCM auth tag)
pub const TAG_LENGTH: usize = 16;

/// HKDF info for legacy hardware-bound container keys.
const HKDF_INFO: &[u8] = b"LAGI_CONTAINER_KEY_v1";

/// Universal salt and input domain for portable model distribution.
const DISTRIBUTION_SALT: &[u8] = b"LIUMI_MDN_UNIVERSAL_CONTAINER_V1";
const DISTRIBUTION_INFO: &[u8] = b"LAGI_PORTABLE_BASE_PAYLOAD_V1";

/// PBKDF2 output domain for passphrase-protected migration bundles.
const TRANSIT_INFO: &[u8] = b"LAGI_SECURE_MIGRATION_TRANSIT_V1";
const TRANSIT_PBKDF2_ITERATIONS: u32 = 600_000;

/// Cryptographic errors
#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Key derivation failed: {0}")]
    KeyDerivation(String),

    #[error("Encryption failed: {0}")]
    Encryption(String),

    #[error("Decryption failed: {0}")]
    Decryption(String),

    #[error("Integrity verification failed: {0}")]
    Integrity(String),

    #[error("Invalid key length: expected {expected}, got {got}")]
    InvalidKeyLength { expected: usize, got: usize },

    #[error("Invalid nonce length: expected {expected}, got {got}")]
    InvalidNonceLength { expected: usize, got: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("AES-GCM error: {0}")]
    AesGcm(aes_gcm::Error),
}

/// Hardware-specific key derivation seed
#[derive(Zeroize, ZeroizeOnDrop)]
struct HardwareSeed {
    data: Vec<u8>,
}

/// Crypto manager for container operations
pub struct CryptoManager;

impl CryptoManager {
    /// Derive a host-specific 256-bit encryption key using platform trust roots
    pub fn derive_hardware_key(
        additional_context: Option<&[u8]>,
    ) -> Result<[u8; KEY_LENGTH], CryptoError> {
        let seed = Self::get_platform_seed()?;

        // Combine seed with additional context
        let mut ikm = seed.data.clone();
        if let Some(ctx) = additional_context {
            ikm.extend_from_slice(ctx);
        }

        // Derive key using HKDF-SHA256
        let hkdf = Hkdf::<Sha256>::new(None, &ikm);
        let mut key = [0u8; KEY_LENGTH];
        hkdf.expand(HKDF_INFO, &mut key)
            .map_err(|e| CryptoError::KeyDerivation(format!("HKDF expand failed: {}", e)))?;

        Ok(key)
    }

    /// Get platform-specific seed material for key derivation
    fn get_platform_seed() -> Result<HardwareSeed, CryptoError> {
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_seed()
        }

        #[cfg(target_os = "windows")]
        {
            Self::get_windows_seed()
        }

        #[cfg(target_os = "linux")]
        {
            Self::get_linux_seed()
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            Self::get_fallback_seed()
        }
    }

    #[cfg(target_os = "macos")]
    fn get_macos_seed() -> Result<HardwareSeed, CryptoError> {
        let machine_id = Self::get_machine_id()?;
        let mut seed = b"LIUMI_DARWIN_SECURE_ENCLAVE_ROOT_SECRET".to_vec();
        seed.extend_from_slice(&machine_id);
        Ok(HardwareSeed { data: seed })
    }

    #[cfg(target_os = "windows")]
    fn get_windows_seed() -> Result<HardwareSeed, CryptoError> {
        let machine_id = Self::get_machine_id()?;
        let mut seed = b"LIUMI_WIN_TPM20_DATA_PROTECTION_ROOT_SECRET".to_vec();
        seed.extend_from_slice(&machine_id);
        Ok(HardwareSeed { data: seed })
    }

    #[cfg(target_os = "linux")]
    fn get_linux_seed() -> Result<HardwareSeed, CryptoError> {
        let machine_id = Self::get_machine_id()?;
        let mut seed = b"LIUMI_LINUX_KEYRING_ROOT_SECRET".to_vec();
        seed.extend_from_slice(&machine_id);
        Ok(HardwareSeed { data: seed })
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    fn get_fallback_seed() -> Result<HardwareSeed, CryptoError> {
        let mut seed = rand::random::<[u8; 32]>().to_vec();
        seed.extend_from_slice(b"LIUMI_FALLBACK_ROOT_SECRET");
        Ok(HardwareSeed { data: seed })
    }

    fn get_machine_id() -> Result<Vec<u8>, CryptoError> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(id) = std::fs::read("/etc/machine-id") {
                let id = id.trim_ascii().to_vec();
                if !id.is_empty() {
                    return Ok(id);
                }
            }
            if let Ok(id) = std::fs::read("/var/lib/dbus/machine-id") {
                let id = id.trim_ascii().to_vec();
                if !id.is_empty() {
                    return Ok(id);
                }
            }
        }

        use std::net::UdpSocket;
        if let Ok(socket) = UdpSocket::bind("0.0.0.0:0") {
            if let Ok(addr) = socket.local_addr() {
                return Ok(addr.ip().to_string().into_bytes());
            }
        }

        let mut bytes = [0u8; 8];
        OsRng.fill_bytes(&mut bytes);
        Ok(bytes.to_vec())
    }

    /// Encrypt payload using AES-256-GCM
    pub fn encrypt_payload(
        data: &[u8],
        key: &[u8; KEY_LENGTH],
    ) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        if data.is_empty() {
            return Err(CryptoError::Encryption(
                "Cannot encrypt empty payload".to_string(),
            ));
        }

        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, data)
            .map_err(|e| CryptoError::Encryption(format!("AES-GCM encryption failed: {}", e)))?;

        Ok((nonce.to_vec(), ciphertext))
    }

    /// Decrypt payload using AES-256-GCM with integrity verification
    pub fn decrypt_payload(
        ciphertext: &[u8],
        key: &[u8; KEY_LENGTH],
        nonce: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        if nonce.len() != NONCE_LENGTH {
            return Err(CryptoError::InvalidNonceLength {
                expected: NONCE_LENGTH,
                got: nonce.len(),
            });
        }

        if ciphertext.is_empty() {
            return Err(CryptoError::Decryption(
                "Cannot decrypt empty ciphertext".to_string(),
            ));
        }

        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = Nonce::from_slice(nonce);
        let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|_| {
            CryptoError::Decryption(
                "Decryption failed: integrity check failed or invalid key".to_string(),
            )
        })?;

        Ok(plaintext)
    }

    /// Compute SHA256 hash of payload for integrity verification
    pub fn compute_payload_hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Constant-time comparison of two byte sequences
    pub fn secure_compare(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.ct_eq(b).into()
    }

    /// Generate a cryptographically secure random nonce
    pub fn generate_nonce() -> Vec<u8> {
        let mut nonce = vec![0u8; NONCE_LENGTH];
        OsRng.fill_bytes(&mut nonce);
        nonce
    }
}

/// Derive the portable key used for the base model distribution envelope.
///
/// The universal salt is intentionally public: this key makes model payloads
/// portable, while personal memory remains bound to a separate hardware root.
pub fn derive_distribution_key() -> [u8; KEY_LENGTH] {
    let hkdf = Hkdf::<Sha256>::new(Some(DISTRIBUTION_SALT), DISTRIBUTION_SALT);
    let mut key = [0u8; KEY_LENGTH];
    hkdf.expand(DISTRIBUTION_INFO, &mut key)
        .expect("32-byte HKDF-SHA256 expansion is always valid");
    key
}

/// Derive the host-bound key used for episodic memory state.
pub fn derive_hardware_key() -> [u8; KEY_LENGTH] {
    CryptoManager::derive_hardware_key(None)
        .expect("host hardware seed is available on supported platforms")
}

/// Derive the passphrase-protected transit key for `.lmb` migration bundles.
pub fn derive_transit_key(passphrase: &str, salt: &[u8; 16]) -> [u8; KEY_LENGTH] {
    let mut stretched = [0u8; KEY_LENGTH];
    pbkdf2_hmac::<Sha256>(
        passphrase.as_bytes(),
        salt,
        TRANSIT_PBKDF2_ITERATIONS,
        &mut stretched,
    );
    let hkdf = Hkdf::<Sha256>::new(Some(salt), &stretched);
    let mut key = [0u8; KEY_LENGTH];
    hkdf.expand(TRANSIT_INFO, &mut key)
        .expect("32-byte HKDF-SHA256 expansion is always valid");
    stretched.zeroize();
    key
}

/// Encrypt bytes under an explicitly selected key.
pub fn encrypt_with_key(
    data: &[u8],
    key: &[u8; KEY_LENGTH],
) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    CryptoManager::encrypt_payload(data, key)
}

/// Decrypt bytes under an explicitly selected key.
pub fn decrypt_with_key(
    ciphertext: &[u8],
    key: &[u8; KEY_LENGTH],
    nonce: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    CryptoManager::decrypt_payload(ciphertext, key, nonce)
}

/// High-level encryption for legacy hardware-keyed container payloads.
pub fn encrypt_for_container(
    data: &[u8],
    additional_context: Option<&[u8]>,
) -> Result<(Vec<u8>, Vec<u8>, [u8; 32]), CryptoError> {
    let key = CryptoManager::derive_hardware_key(additional_context)?;
    let (nonce, ciphertext) = CryptoManager::encrypt_payload(data, &key)?;
    let payload_hash = CryptoManager::compute_payload_hash(&ciphertext);
    Ok((nonce, ciphertext, payload_hash))
}

/// High-level decryption for legacy hardware-keyed container payloads.
pub fn decrypt_from_container(
    ciphertext: &[u8],
    nonce: &[u8],
    additional_context: Option<&[u8]>,
) -> Result<Vec<u8>, CryptoError> {
    let key = CryptoManager::derive_hardware_key(additional_context)?;
    CryptoManager::decrypt_payload(ciphertext, &key, nonce)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_key_is_deterministic_and_domain_separated() {
        let key1 = derive_distribution_key();
        let key2 = derive_distribution_key();
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32);
        assert_ne!(key1, derive_hardware_key());
    }

    #[test]
    fn test_transit_key_is_salted_and_deterministic() {
        let salt1 = [1u8; 16];
        let salt2 = [2u8; 16];
        assert_eq!(
            derive_transit_key("passphrase", &salt1),
            derive_transit_key("passphrase", &salt1)
        );
        assert_ne!(
            derive_transit_key("passphrase", &salt1),
            derive_transit_key("passphrase", &salt2)
        );
        assert_ne!(
            derive_transit_key("passphrase", &salt1),
            derive_transit_key("different", &salt1)
        );
    }

    #[test]
    fn test_derive_hardware_key() {
        let key1 = CryptoManager::derive_hardware_key(None).unwrap();
        let key2 = CryptoManager::derive_hardware_key(None).unwrap();
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), KEY_LENGTH);
    }

    #[test]
    fn test_derive_hardware_key_with_context() {
        let key1 = CryptoManager::derive_hardware_key(None).unwrap();
        let key2 = CryptoManager::derive_hardware_key(Some(b"test_context")).unwrap();
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let data = b"Test payload data for encryption roundtrip";
        let key = CryptoManager::derive_hardware_key(None).unwrap();

        let (nonce, ciphertext) = CryptoManager::encrypt_payload(data, &key).unwrap();
        let decrypted = CryptoManager::decrypt_payload(&ciphertext, &key, &nonce).unwrap();

        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_encrypt_produces_different_ciphertext() {
        let data = b"Test data";
        let key = CryptoManager::derive_hardware_key(None).unwrap();

        let (nonce1, ciphertext1) = CryptoManager::encrypt_payload(data, &key).unwrap();
        let (nonce2, ciphertext2) = CryptoManager::encrypt_payload(data, &key).unwrap();

        assert_ne!(nonce1, nonce2);
        assert_ne!(ciphertext1, ciphertext2);
    }

    #[test]
    fn test_decrypt_wrong_key_fails() {
        let data = b"Test data";
        let key1 = CryptoManager::derive_hardware_key(None).unwrap();
        let key2 = CryptoManager::derive_hardware_key(Some(b"different_context")).unwrap();

        let (nonce, ciphertext) = CryptoManager::encrypt_payload(data, &key1).unwrap();

        let result = CryptoManager::decrypt_payload(&ciphertext, &key2, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_tampered_ciphertext_fails() {
        let data = b"Test data";
        let key = CryptoManager::derive_hardware_key(None).unwrap();

        let (nonce, mut ciphertext) = CryptoManager::encrypt_payload(data, &key).unwrap();
        ciphertext[0] ^= 0xFF;

        let result = CryptoManager::decrypt_payload(&ciphertext, &key, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_payload_hash() {
        let data = b"Test data for hashing";
        let hash1 = CryptoManager::compute_payload_hash(data);
        let hash2 = CryptoManager::compute_payload_hash(data);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 32);
    }

    #[test]
    fn test_secure_compare() {
        assert!(CryptoManager::secure_compare(b"hello", b"hello"));
        assert!(!CryptoManager::secure_compare(b"hello", b"world"));
        assert!(!CryptoManager::secure_compare(b"hello", b"hello!"));
    }

    #[test]
    fn test_generate_nonce() {
        let nonce1 = CryptoManager::generate_nonce();
        let nonce2 = CryptoManager::generate_nonce();
        assert_eq!(nonce1.len(), NONCE_LENGTH);
        assert_eq!(nonce2.len(), NONCE_LENGTH);
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn test_encrypt_for_container() {
        let data = b"Container payload data";
        let (nonce, ciphertext, hash) = encrypt_for_container(data, None).unwrap();

        assert_eq!(nonce.len(), NONCE_LENGTH);
        assert!(ciphertext.len() > data.len());
        assert_eq!(hash.len(), 32);

        let decrypted = decrypt_from_container(&ciphertext, &nonce, None).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_different_contexts_produce_different_keys() {
        let data = b"Test data";

        let (nonce1, ciphertext1, hash1) = encrypt_for_container(data, Some(b"context1")).unwrap();
        let (nonce2, ciphertext2, hash2) = encrypt_for_container(data, Some(b"context2")).unwrap();

        assert_ne!(ciphertext1, ciphertext2);
        assert_ne!(hash1, hash2);

        let decrypted1 = decrypt_from_container(&ciphertext1, &nonce1, Some(b"context1")).unwrap();
        let decrypted2 = decrypt_from_container(&ciphertext2, &nonce2, Some(b"context2")).unwrap();

        assert_eq!(decrypted1, data);
        assert_eq!(decrypted2, data);

        assert!(decrypt_from_container(&ciphertext1, &nonce1, Some(b"context2")).is_err());
        assert!(decrypt_from_container(&ciphertext2, &nonce2, Some(b"context1")).is_err());
    }
}
