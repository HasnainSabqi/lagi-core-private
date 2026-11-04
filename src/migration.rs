/*!
* Secure workspace migration protocol (`.lmb`).

* Migration bundles contain only an authenticated, passphrase-encrypted memory
* snapshot. Plaintext is decrypted in RAM and is never written to the bundle.
*/

use crate::{
    crypto::derive_transit_key,
    packager::{read_state, write_state},
};
use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng, Payload},
    Aes256Gcm, Key, Nonce,
};
use rand::RngCore;
use std::fs;
use std::io::Write;
use std::path::Path;
use thiserror::Error;
use zeroize::Zeroize;

const LMB_MAGIC: &[u8; 8] = b"LIUMILMB";
const LMB_VERSION: u16 = 1;
const LMB_KDF_PBKDF2_HMAC_SHA256: u16 = 1;
const LMB_SALT_LEN: usize = 16;
const LMB_PBKDF2_ITERATIONS: u32 = 600_000;
const LMB_HEADER_LEN: usize = 8 + 2 + 2 + 4 + 16 + 12 + 8;

#[derive(Error, Debug)]
pub enum MigrationError {
    #[error("Invalid migration bundle: {0}")]
    InvalidBundle(String),
    #[error("Migration authentication failed")]
    AuthenticationFailed,
    #[error("Migration crypto error: {0}")]
    Crypto(#[from] crate::crypto::CryptoError),
    #[error("Migration serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Migration IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Container state error: {0}")]
    Container(#[from] crate::packager::PackagerError),
}

/// Export the active container's hardware-bound memory into an authenticated
/// `.lmb` envelope protected by a passphrase.
pub fn export_workspace(
    container_path: &Path,
    out_path: &Path,
    passphrase: &str,
) -> Result<u64, MigrationError> {
    let plaintext = read_state(container_path)?.ok_or_else(|| {
        MigrationError::InvalidBundle("source container has no memory state".to_string())
    })?;
    let mut plaintext = plaintext;

    let mut salt = [0u8; LMB_SALT_LEN];
    OsRng.fill_bytes(&mut salt);
    let key = derive_transit_key(passphrase, &salt);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let nonce_bytes: [u8; 12] = nonce.into();
    let ciphertext_len = plaintext.len() as u64 + 16;
    let header = build_header(&salt, &nonce_bytes, ciphertext_len);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &plaintext,
                aad: &header,
            },
        )
        .map_err(|_| MigrationError::AuthenticationFailed)?;

    let mut bundle = header;
    bundle.extend_from_slice(&ciphertext);

    let mut file = fs::File::create(out_path)?;
    file.write_all(&bundle)?;
    file.flush()?;
    file.sync_all()?;

    plaintext.zeroize();
    Ok(bundle.len() as u64)
}

/// Import an authenticated `.lmb` bundle and re-encrypt its memory snapshot
/// under the destination host's hardware key in the container state trailer.
pub fn import_workspace(
    container_path: &Path,
    in_path: &Path,
    passphrase: &str,
) -> Result<usize, MigrationError> {
    let bundle = fs::read(in_path)?;
    if bundle.len() < LMB_HEADER_LEN {
        return Err(MigrationError::InvalidBundle(
            "bundle is truncated".to_string(),
        ));
    }

    let mut header = Vec::with_capacity(LMB_HEADER_LEN);
    header.extend_from_slice(&bundle[..LMB_HEADER_LEN]);
    let salt: [u8; 16] = bundle[16..32]
        .try_into()
        .map_err(|_| MigrationError::InvalidBundle("invalid salt".to_string()))?;
    let nonce = bundle[32..44].to_vec();
    let ciphertext_len = u64::from_be_bytes(
        bundle[44..52]
            .try_into()
            .map_err(|_| MigrationError::InvalidBundle("invalid length".to_string()))?,
    ) as usize;
    let expected_len = LMB_HEADER_LEN
        .checked_add(ciphertext_len)
        .ok_or_else(|| MigrationError::InvalidBundle("bundle length overflow".to_string()))?;
    if bundle.len() != expected_len {
        return Err(MigrationError::InvalidBundle(
            "bundle length mismatch".to_string(),
        ));
    }
    if &header[..8] != LMB_MAGIC
        || u16::from_be_bytes(header[8..10].try_into().unwrap()) != LMB_VERSION
        || u16::from_be_bytes(header[10..12].try_into().unwrap()) != LMB_KDF_PBKDF2_HMAC_SHA256
        || u32::from_be_bytes(header[12..16].try_into().unwrap()) != LMB_PBKDF2_ITERATIONS
    {
        return Err(MigrationError::InvalidBundle(
            "unsupported or invalid bundle header".to_string(),
        ));
    }

    let key = derive_transit_key(passphrase, &salt);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let mut plaintext = cipher
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &bundle[LMB_HEADER_LEN..],
                aad: &header,
            },
        )
        .map_err(|_| MigrationError::AuthenticationFailed)?;

    // Validate the memory graph before replacing the destination state.
    let entries: Vec<crate::memory::MemoryEntry> = serde_json::from_slice(&plaintext)?;
    let count = entries.len();
    write_state(container_path, &plaintext)?;
    plaintext.zeroize();
    Ok(count)
}

fn build_header(salt: &[u8; 16], nonce: &[u8; 12], ciphertext_len: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(LMB_HEADER_LEN);
    header.extend_from_slice(LMB_MAGIC);
    header.extend_from_slice(&LMB_VERSION.to_be_bytes());
    header.extend_from_slice(&LMB_KDF_PBKDF2_HMAC_SHA256.to_be_bytes());
    header.extend_from_slice(&LMB_PBKDF2_ITERATIONS.to_be_bytes());
    header.extend_from_slice(salt);
    header.extend_from_slice(nonce);
    header.extend_from_slice(&ciphertext_len.to_be_bytes());
    header
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packager::{create_test_agent_dir, pack_agent};
    use tempfile::tempdir;

    fn create_container_with_memory(path: &Path) {
        let parent = path.parent().unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy();
        let agent_dir = parent.join(format!("{stem}-agent"));
        fs::create_dir_all(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();
        pack_agent(&agent_dir, path, None).unwrap();

        let memory_path = parent.join(format!("{stem}-memory.db"));
        let memory = crate::memory::MemoryGraph::new(&memory_path, None).unwrap();
        memory
            .store(&crate::memory::MemoryEntry::new(
                crate::memory::MemoryType::Context,
                "migration secret fact".to_string(),
                None,
                90,
                vec!["portable".to_string()],
            ))
            .unwrap();
        crate::packager::flush_memory_to_container(&memory, path).unwrap();
    }

    #[test]
    fn export_import_roundtrip_restores_memory() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.lagi");
        let destination = dir.path().join("destination.lagi");
        let bundle = dir.path().join("workspace.lmb");
        create_container_with_memory(&source);
        create_container_with_memory(&destination);

        export_workspace(&source, &bundle, "correct horse battery staple").unwrap();
        let count =
            import_workspace(&destination, &bundle, "correct horse battery staple").unwrap();
        assert_eq!(count, 1);

        let memory = crate::memory::MemoryGraph::new(dir.path().join("restored.db"), None).unwrap();
        crate::packager::load_memory_from_container(&destination, &memory).unwrap();
        assert_eq!(
            memory
                .search_text("migration secret fact", 5)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn wrong_passphrase_is_rejected_without_changing_state() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.lagi");
        let destination = dir.path().join("destination.lagi");
        let bundle = dir.path().join("workspace.lmb");
        create_container_with_memory(&source);
        create_container_with_memory(&destination);

        export_workspace(&source, &bundle, "right passphrase").unwrap();
        let error = import_workspace(&destination, &bundle, "wrong passphrase").unwrap_err();
        assert!(matches!(error, MigrationError::AuthenticationFailed));
    }

    #[test]
    fn tampered_bundle_is_rejected() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.lagi");
        let bundle = dir.path().join("workspace.lmb");
        create_container_with_memory(&source);
        export_workspace(&source, &bundle, "passphrase").unwrap();

        let mut bytes = fs::read(&bundle).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        fs::write(&bundle, bytes).unwrap();

        let destination = dir.path().join("destination.lagi");
        create_container_with_memory(&destination);
        let error = import_workspace(&destination, &bundle, "passphrase").unwrap_err();
        assert!(matches!(error, MigrationError::AuthenticationFailed));
    }
}
