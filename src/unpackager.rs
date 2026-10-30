/*!
RAM-Only Unpackager & Integrity Verification

Unpacks .lagi containers entirely in memory without writing decrypted assets to disk.
Validates AES-256-GCM authentication tags and SHA256 payload checksums.
*/

use crate::{
    crypto::{decrypt_from_container, CryptoManager, CryptoError},
    manifest::{LagiManifest, ManifestError},
    spec::{HEADER_SIZE, LagiHeader, verify_magic, EncryptionMode, SpecError},
};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{Read, Cursor};
use std::path::{Path, PathBuf};
use tar::Archive;
use thiserror::Error;

/// Unpackager errors
#[derive(Error, Debug)]
pub enum UnpackagerError {
    #[error("Invalid container: {0}")]
    InvalidContainer(String),

    #[error("Crypto error: {0}")]
    Crypto(#[from] CryptoError),

    #[error("Manifest error: {0}")]
    Manifest(#[from] ManifestError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Spec error: {0}")]
    Spec(#[from] SpecError),

    #[error("Integrity verification failed: {0}")]
    IntegrityVerification(String),

    #[error("Unsupported encryption mode: {0}")]
    UnsupportedEncryptionMode(u16),
}

/// In-memory asset extracted from container
#[derive(Debug, Clone)]
pub struct Asset {
    pub name: String,
    pub data: Vec<u8>,
}

/// Unpacked container with manifest and in-memory assets
#[derive(Debug)]
pub struct UnpackedContainer {
    pub manifest: LagiManifest,
    pub assets: Vec<Asset>,
    pub header: LagiHeader,
}

impl UnpackedContainer {
    /// Get asset by name
    pub fn get_asset(&self, name: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.name == name)
    }

    /// Get model GGUF data (resolved via either manifest schema)
    pub fn get_model(&self) -> Option<&[u8]> {
        let name = self.manifest.model_gguf_filename()?;
        self.get_asset(name).map(|a| a.data.as_slice())
    }
}

/// LagiUnpackager - decrypts and validates .lagi containers in RAM
pub struct LagiUnpackager {
    path: PathBuf,
}

impl LagiUnpackager {
    /// Create a new unpackager for the given .lagi file
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self { path: path.as_ref().to_path_buf() }
    }

    /// Unpack and validate container entirely in RAM
    pub fn unpack(&self, additional_context: Option<&[u8]>) -> Result<UnpackedContainer, UnpackagerError> {
        let mut file = File::open(&self.path)?;
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;

        let header = LagiHeader::from_bytes(&header_bytes)?;

        if !verify_magic(&header.magic) {
            return Err(UnpackagerError::InvalidContainer("Invalid magic bytes".to_string()));
        }

        if header.version != crate::spec::FORMAT_VERSION {
            return Err(UnpackagerError::InvalidContainer(format!(
                "Unsupported format version: {}", header.version
            )));
        }

        if header.enc_mode != EncryptionMode::Aes256GcmHardware {
            return Err(UnpackagerError::UnsupportedEncryptionMode(header.enc_mode.to_u16()));
        }

        let mut manifest_bytes = vec![0u8; header.manifest_len as usize];
        file.read_exact(&mut manifest_bytes)?;
        let manifest = LagiManifest::from_json_bytes(&manifest_bytes)?;

        // The trailing dynamic-state block (LVCST01), if present, is NOT part
        // of the encrypted payload: read only up to the block start.
        let file_len = file.metadata()?.len();
        let payload_end = crate::packager::find_state_block(&self.path)
            .map_err(|e| {
                UnpackagerError::InvalidContainer(format!("state block scan failed: {}", e))
            })?
            .map(|b| file_len - b.total_len())
            .unwrap_or(file_len);
        let payload_len =
            (payload_end - (HEADER_SIZE + header.manifest_len as usize) as u64) as usize;
        let mut encrypted_payload = vec![0u8; payload_len];
        file.read_exact(&mut encrypted_payload)?;

        let expected_hash = header.payload_hash;
        let actual_hash = CryptoManager::compute_payload_hash(&encrypted_payload);
        if !CryptoManager::secure_compare(&expected_hash, &actual_hash) {
            return Err(UnpackagerError::IntegrityVerification(
                "Payload SHA256 checksum mismatch".to_string()
            ));
        }

        let decrypted_payload = decrypt_from_container(&encrypted_payload, &header.nonce, additional_context)?;

        let assets = Self::extract_assets_from_tarball(&decrypted_payload)?;

        let model_name = manifest
            .model_gguf_filename()
            .ok_or_else(|| {
                UnpackagerError::InvalidContainer("Manifest declares no model file".to_string())
            })?
            .to_string();

        if !assets.iter().any(|a| a.name == model_name) {
            return Err(UnpackagerError::InvalidContainer(
                format!("Model file '{}' not found in container", model_name),
            ));
        }

        Ok(UnpackedContainer { manifest, assets, header })
    }
/// Extract assets from gzipped tarball in memory
    fn extract_assets_from_tarball(data: &[u8]) -> Result<Vec<Asset>, UnpackagerError> {
        let mut assets = Vec::new();
        let decoder = GzDecoder::new(Cursor::new(data));
        let mut archive = Archive::new(decoder);

        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().to_string();
            
            if entry.header().entry_type().is_file() {
                let mut data = Vec::new();
                entry.read_to_end(&mut data)?;
                assets.push(Asset { name: path, data });
            }
        }

        Ok(assets)
    }

    /// Inspect container metadata without full unpacking
    pub fn inspect(&self) -> Result<ContainerInfo, UnpackagerError> {
        let mut file = File::open(&self.path)?;
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;

        let header = LagiHeader::from_bytes(&header_bytes)?;

        if !verify_magic(&header.magic) {
            return Err(UnpackagerError::InvalidContainer("Invalid magic bytes".to_string()));
        }

        let mut manifest_bytes = vec![0u8; header.manifest_len as usize];
        file.read_exact(&mut manifest_bytes)?;
        let manifest = LagiManifest::from_json_bytes(&manifest_bytes)?;

        Ok(ContainerInfo {
            magic: String::from_utf8_lossy(&header.magic).to_string(),
            version: header.version,
            encryption_mode: header.enc_mode.to_u16(),
            manifest_length: header.manifest_len,
            reserved: header.reserved,
            nonce_hex: hex::encode(header.nonce),
            payload_hash_hex: hex::encode(header.payload_hash),
            manifest,
            file_size: self.path.metadata()?.len(),
        })
    }
}

/// Container metadata for inspection
#[derive(Debug, serde::Serialize)]
pub struct ContainerInfo {
    pub magic: String,
    pub version: u16,
    pub encryption_mode: u16,
    pub manifest_length: u32,
    pub reserved: u32,
    pub nonce_hex: String,
    pub payload_hash_hex: String,
    pub manifest: LagiManifest,
    pub file_size: u64,
}

/// Convenience function to unpack an agent
pub fn unpack_agent<P: AsRef<Path>>(
    path: P,
    additional_context: Option<&[u8]>,
) -> Result<UnpackedContainer, UnpackagerError> {
    let unpackager = LagiUnpackager::new(path);
    unpackager.unpack(additional_context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packager::{create_test_agent_dir, pack_agent};
    use tempfile::tempdir;

    #[test]
    fn test_unpack_valid_container() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        std::fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();

        let output_path = dir.path().join("test.lagi");
        pack_agent(&agent_dir, &output_path, None).unwrap();

        let unpackager = LagiUnpackager::new(&output_path);
        let unpacked = unpackager.unpack(None).unwrap();

        assert_eq!(unpacked.manifest.name, "TestAgent");
        assert!(unpacked.get_model().is_some());
    }

    #[test]
    fn test_unpack_rejects_invalid_magic() {
        let dir = tempdir().unwrap();
        let bad_file = dir.path().join("bad.lagi");
        let mut bad_data = b"INVALID_MAGIC".to_vec();
        bad_data.extend_from_slice(&[0u8; 100]);
        std::fs::write(&bad_file, bad_data).unwrap();

        let unpackager = LagiUnpackager::new(&bad_file);
        let result = unpackager.unpack(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_unpack_rejects_tampered_payload() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        std::fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();

        let output_path = dir.path().join("test.lagi");
        pack_agent(&agent_dir, &output_path, None).unwrap();

        let mut file = std::fs::OpenOptions::new().read(true).write(true).open(&output_path).unwrap();
        use std::io::{Seek, Write};
        // Tamper with the trailing encrypted payload so the SHA256 integrity
        // check fails with IntegrityVerification before decryption
        let file_len = std::fs::metadata(&output_path).unwrap().len();
        file.seek(std::io::SeekFrom::Start(file_len - 8)).unwrap();
        file.write_all(b"TAMPERED").unwrap();

        let unpackager = LagiUnpackager::new(&output_path);
        let result = unpackager.unpack(None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), UnpackagerError::IntegrityVerification(_)));
    }

    #[test]
    fn test_inspect_reads_header_only() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        std::fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();

        let output_path = dir.path().join("test.lagi");
        pack_agent(&agent_dir, &output_path, None).unwrap();

        let unpackager = LagiUnpackager::new(&output_path);
        let info = unpackager.inspect().unwrap();

        assert_eq!(info.magic, "LAGI_V01");
        assert_eq!(info.version, 1);
        assert_eq!(info.encryption_mode, 1);
        assert_eq!(info.manifest.name, "TestAgent");
    }
}