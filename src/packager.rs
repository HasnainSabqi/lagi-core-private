/*!
Packaging Engine - .lagi File Generator

Compiles agent source directory into an encrypted .lagi binary container.
*/

use crate::{
    crypto::{encrypt_for_container, CryptoError},
    manifest::{LagiManifest, ManifestError},
    spec::{MAGIC_BYTES, FORMAT_VERSION, EncryptionMode, pack_header, MAX_MANIFEST_SIZE},
};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tar::Builder;
use walkdir::WalkDir;
use thiserror::Error;

/// Packager errors
#[derive(Error, Debug)]
pub enum PackagerError {
    #[error("Manifest validation failed: {0}")]
    ManifestValidation(#[from] ManifestError),
    
    #[error("Crypto error: {0}")]
    Crypto(#[from] CryptoError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Manifest too large: {0} bytes (max {})", MAX_MANIFEST_SIZE)]
    ManifestTooLarge(usize),
    
    #[error("Source directory not found: {0}")]
    SourceNotFound(String),
    
    #[error("Invalid source: not a directory: {0}")]
    NotADirectory(String),
}

/// Create a gzipped tarball of all assets except manifest.json
fn create_payload_tarball(source_dir: &Path) -> Result<Vec<u8>, PackagerError> {
    let mut tar_data = Vec::new();
    {
        let encoder = GzEncoder::new(&mut tar_data, Compression::default());
        let mut builder = Builder::new(encoder);
        
        for entry in WalkDir::new(source_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            
            // Skip manifest.json
            if path.file_name().and_then(|n| n.to_str()) == Some("manifest.json") {
                continue;
            }
            
            // Skip directories (they're added implicitly)
            if path.is_dir() {
                continue;
            }
            
            let relative_path = path.strip_prefix(source_dir)
                .map_err(|_| PackagerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Failed to strip source prefix"
                )))?;
            
            let mut file = fs::File::open(path)?;
            builder.append_file(relative_path, &mut file)?;
        }
        
        builder.finish()?;
    }
    
    Ok(tar_data)
}

/// LagiPackager - compiles source directory into .lagi container
pub struct LagiPackager {
    source_dir: PathBuf,
}

impl LagiPackager {
    /// Create a new packager for the given source directory
    pub fn new<P: AsRef<Path>>(source_dir: P) -> Result<Self, PackagerError> {
        let source_dir = source_dir.as_ref().to_path_buf();
        
        if !source_dir.exists() {
            return Err(PackagerError::SourceNotFound(source_dir.display().to_string()));
        }
        
        if !source_dir.is_dir() {
            return Err(PackagerError::NotADirectory(source_dir.display().to_string()));
        }
        
        Ok(LagiPackager { source_dir })
    }
    
    /// Load and validate manifest from source directory
    fn load_manifest(&self) -> Result<LagiManifest, PackagerError> {
        let manifest_path = self.source_dir.join("manifest.json");
        let manifest = LagiManifest::from_file(&manifest_path)?;
        Ok(manifest)
    }
    
    /// Package source directory into .lagi container
    pub fn pack<P: AsRef<Path>>(&self, output_path: P, additional_context: Option<&[u8]>) -> Result<(), PackagerError> {
        let output_path = output_path.as_ref();
        
        // Load and validate manifest
        let manifest = self.load_manifest()?;
        let manifest_bytes = manifest.to_json_bytes()?;
        
        // Verify manifest size
        if manifest_bytes.len() > MAX_MANIFEST_SIZE {
            return Err(PackagerError::ManifestTooLarge(manifest_bytes.len()));
        }
        
        // Create payload tarball
        let raw_payload = create_payload_tarball(&self.source_dir)?;
        
        // Encrypt payload
        let (nonce, encrypted_payload, payload_hash) = encrypt_for_container(&raw_payload, additional_context)?;
        
        // Pack header
        let header = pack_header(
            MAGIC_BYTES,
            FORMAT_VERSION,
            EncryptionMode::Aes256GcmHardware,
            manifest_bytes.len() as u32,
            0, // reserved
            &nonce.try_into().map_err(|_| PackagerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid nonce length"
            )))?,
            &payload_hash,
        );
        
        // Write container
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let mut file = fs::File::create(output_path)?;
        file.write_all(&header)?;
        file.write_all(&manifest_bytes)?;
        file.write_all(&encrypted_payload)?;
        
        println!("Packaged {} -> {}", self.source_dir.display(), output_path.display());
        println!("  Manifest: {} bytes", manifest_bytes.len());
        println!("  Payload: {} bytes (raw) -> {} bytes (encrypted)", raw_payload.len(), encrypted_payload.len());
        println!("  Encryption: AES-256-GCM (hardware-derived key)");
        
        Ok(())
    }
}

/// Convenience function to package an agent
pub fn pack_agent<P: AsRef<Path>, Q: AsRef<Path>>(
    source_dir: P,
    output_path: Q,
    additional_context: Option<&[u8]>,
) -> Result<(), PackagerError> {
    let packager = LagiPackager::new(source_dir)?;
    packager.pack(output_path, additional_context)
}

// ---------------------------------------------------------------------------
// Single-file container state persistence (trailing .lvc state block)
// ---------------------------------------------------------------------------

/// Magic bytes of the trailing dynamic-state block inside a `.lagi` file.
pub const STATE_MAGIC: &[u8; 8] = b"LVCSTATE";

/// Wire format of the trailing state block:
/// `magic(8) | nonce_len(u32 BE) | nonce | ciphertext_len(u64 BE) | ciphertext`
/// The plaintext is an AES-256-GCM encrypted, hardware-key-sealed snapshot of
/// the Liumi VectorCore (`.lvc`) memory graph (serde_json of Vec<MemoryEntry>).
#[derive(Debug, Clone)]
pub struct RawStateBlock {
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

impl RawStateBlock {
    pub fn total_len(&self) -> u64 {
        (8 + 4 + self.nonce.len() + 8 + self.ciphertext.len()) as u64
    }
}

/// Locate and parse the trailing state block of a `.lagi` file, if present.
pub fn find_state_block(path: &Path) -> Result<Option<RawStateBlock>, PackagerError> {
    use byteorder::{BigEndian, ReadBytesExt};
    use std::io::{Read, Seek, SeekFrom};

    let file_len = fs::metadata(path)?.len();
    if file_len < 8 {
        return Ok(None);
    }
    let mut file = fs::File::open(path)?;

    // The block has variable length, so scan backwards in 4KB windows until
    // STATE_MAGIC is found (overlapping windows so a split magic is seen).
    const WINDOW: u64 = 4096;
    let mut search_end = file_len;
    let block_start = loop {
        let start = search_end.saturating_sub(WINDOW);
        let len = (search_end - start) as usize;
        file.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        if let Some(pos) = buf.windows(8).position(|w| w == STATE_MAGIC) {
            break start + pos as u64;
        }
        if start == 0 {
            return Ok(None);
        }
        search_end = start + 8;
    };

    file.seek(SeekFrom::Start(block_start))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != STATE_MAGIC {
        return Ok(None);
    }
    let nonce_len = file.read_u32::<BigEndian>()? as usize;
    let mut nonce = vec![0u8; nonce_len];
    file.read_exact(&mut nonce)?;
    let ct_len = file.read_u64::<BigEndian>()? as usize;
    let mut ciphertext = vec![0u8; ct_len];
    file.read_exact(&mut ciphertext)?;
    Ok(Some(RawStateBlock { nonce, ciphertext }))
}

/// Read and decrypt the `.lvc` state snapshot from a `.lagi` container.
/// Returns `Ok(None)` when the container carries no state block.
pub fn read_state(path: &Path) -> Result<Option<Vec<u8>>, PackagerError> {
    match find_state_block(path)? {
        None => Ok(None),
        Some(block) => Ok(Some(crate::crypto::decrypt_from_container(
            &block.ciphertext,
            &block.nonce,
            None,
        )?)),
    }
}

/// Serialize (encrypt + seal) a `.lvc` state snapshot and write it as the
/// trailing block of the `.lagi` container — **without** rewriting the model
/// payload. The payload bytes are never touched: the old state block (if any)
/// is truncated away and the new one is appended at the same offset.
///
/// Returns the number of bytes written for the new block.
pub fn write_state(path: &Path, plaintext_state: &[u8]) -> Result<u64, PackagerError> {
    use std::io::{Seek, Write};

    let (nonce, ciphertext, _hash) =
        crate::crypto::encrypt_for_container(plaintext_state, None).map_err(PackagerError::Crypto)?;
    let block = RawStateBlock { nonce, ciphertext };

    // Open the container once in read+write mode so the file handle and the
    // length we operate on are always consistent.
    let mut file = fs::OpenOptions::new().read(true).write(true).open(path)?;

    // Explicitly check the existing file length before doing any in-place
    // seek/write. A missing or empty container is rejected instead of being
    // turned into a corrupt file.
    let file_len = file.metadata()?.len();
    if file_len == 0 {
        return Err(io_error("cannot write state to an empty container")(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "container file is empty",
        )));
    }

    let payload_end = match find_state_block(path)? {
        Some(old) => file_len - old.total_len(),
        None => file_len,
    };
    if payload_end > file_len {
        return Err(io_error("invalid state block offset")(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "state block offset exceeds container length",
        )));
    }

    let mut out: Vec<u8> = Vec::with_capacity(block.total_len() as usize);
    out.extend_from_slice(STATE_MAGIC);
    out.extend_from_slice(&(block.nonce.len() as u32).to_be_bytes());
    out.extend_from_slice(&block.nonce);
    out.extend_from_slice(&(block.ciphertext.len() as u64).to_be_bytes());
    out.extend_from_slice(&block.ciphertext);

    // Seek to the state offset and write the LVCSTATE block. The length is
    // adjusted only after the new bytes are written, so a failed write never
    // leaves a truncated container behind.
    file.seek(std::io::SeekFrom::Start(payload_end))?;
    file.write_all(&out)?;
    file.set_len(payload_end + out.len() as u64)?;
    file.flush()?;
    // Force all buffered data and metadata out to NTFS before closing. Without
    // this, exiting right after the in-place seek/write can leave the file
    // corrupt (NTFS error 1392).
    file.sync_all()?;
    Ok(out.len() as u64)
}

/// Export a live [`crate::memory::MemoryGraph`] into a `.lagi` container's
/// trailing state block (encrypts the serde_json snapshot of all entries).
pub fn flush_memory_to_container(
    memory: &crate::memory::MemoryGraph,
    container_path: &Path,
) -> Result<u64, PackagerError> {
    let entries = memory
        .export_state()
        .map_err(io_error("memory export failed"))?;
    let json =
        serde_json::to_vec(&entries).map_err(io_error("memory state serialization failed"))?;
    write_state(container_path, &json)
}

/// Load a `.lvc` state snapshot from a `.lagi` container into a live
/// [`crate::memory::MemoryGraph`]. Returns the number of merged entries.
pub fn load_memory_from_container(
    container_path: &Path,
    memory: &crate::memory::MemoryGraph,
) -> Result<usize, PackagerError> {
    match read_state(container_path)? {
        None => Ok(0),
        Some(plaintext) => {
            let entries: Vec<crate::memory::MemoryEntry> =
                serde_json::from_slice(&plaintext).map_err(io_error("state parse failed"))?;
            memory
                .import_state(&entries)
                .map_err(io_error("memory import failed"))
        }
    }
}

fn io_error<E: std::fmt::Display>(msg: &'static str) -> impl Fn(E) -> PackagerError {
    move |e| PackagerError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{}: {}", msg, e)))
}

#[cfg(test)]
pub(crate) fn create_test_agent_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = LagiManifest {
        lagi_version: "1.0.0".to_string(),
        name: "TestAgent".to_string(),
        author: "Test".to_string(),
        description: "Test agent for unit testing".to_string(),
        privacy_level: "AIR_GAPPED_LOCAL".to_string(),
        model: Some(crate::manifest::ModelConfig {
            series: "Test-Model-1B".to_string(),
            gguf_file: "model.gguf".to_string(),
            quantization: "Q4_K_M".to_string(),
            context_window: 2048,
        }),
        tools: vec![],
        ideation: None,
        model_config: None,
        ideation_loop: None,
        memory_config: None,
        env_vars: std::collections::HashMap::new(),
    };

    manifest.to_file(&dir.join("manifest.json"))?;
    fs::write(dir.join("model.gguf"), b"DUMMY_GGUF_MODEL_DATA")?;
    fs::create_dir(dir.join("prompts"))?;
    fs::write(dir.join("prompts/system.txt"), "You are a test agent.")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::HEADER_SIZE;
    use crate::unpackager::LagiUnpackager;
    use std::io::Read;
    use tempfile::tempdir;

    #[test]
    fn test_pack_creates_valid_container() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();
        
        let output_path = dir.path().join("test.lagi");
        let packager = LagiPackager::new(&agent_dir).unwrap();
        packager.pack(&output_path, None).unwrap();
        
        assert!(output_path.exists());
        assert!(output_path.metadata().unwrap().len() > HEADER_SIZE as u64);
    }
    
    #[test]
    fn test_pack_creates_correct_header() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();
        
        let output_path = dir.path().join("test.lagi");
        let packager = LagiPackager::new(&agent_dir).unwrap();
        packager.pack(&output_path, None).unwrap();
        
        let mut file = fs::File::open(&output_path).unwrap();
        let mut header = vec![0u8; HEADER_SIZE];
        file.read_exact(&mut header).unwrap();
        
        assert_eq!(&header[0..8], MAGIC_BYTES);
    }
    
    #[test]
    fn test_pack_rejects_missing_manifest() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("no_manifest");
        fs::create_dir(&agent_dir).unwrap();
        fs::write(agent_dir.join("model.gguf"), b"dummy").unwrap();
        
        let packager = LagiPackager::new(&agent_dir).unwrap();
        let output_path = dir.path().join("test.lagi");
        
        let result = packager.pack(&output_path, None);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_pack_with_additional_context() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();
        
        let output_path = dir.path().join("test.lagi");
        let packager = LagiPackager::new(&agent_dir).unwrap();
        packager.pack(&output_path, Some(b"agent_123")).unwrap();
        
        assert!(output_path.exists());
    }

    #[test]
    fn test_state_block_roundtrip_and_inplace_update() {
        let dir = tempdir().unwrap();
        let agent_dir = dir.path().join("test_agent");
        fs::create_dir(&agent_dir).unwrap();
        create_test_agent_dir(&agent_dir).unwrap();

        let output_path = dir.path().join("test.lagi");
        pack_agent(&agent_dir, &output_path, None).unwrap();
        let base_size = fs::metadata(&output_path).unwrap().len();

        // No state block initially; unpack works.
        assert!(find_state_block(&output_path).unwrap().is_none());
        assert!(LagiUnpackager::new(&output_path).unpack(None).is_ok());

        // Flush a memory snapshot into the container.
        let mem_path = dir.path().join("lvc.db");
        let memory = crate::memory::MemoryGraph::new(&mem_path, None).unwrap();
        let entry = crate::memory::MemoryEntry::new(
            crate::memory::MemoryType::Context,
            "Tokamaks confine plasma with magnetic fields.".to_string(),
            None,
            80,
            vec!["auto-taught".to_string()],
        );
        memory.store(&entry).unwrap();
        flush_memory_to_container(&memory, &output_path).unwrap();

        assert!(find_state_block(&output_path).unwrap().is_some());
        let grew = fs::metadata(&output_path).unwrap().len();
        assert!(grew > base_size);

        // Model payload untouched: unpack still validates.
        let unpacked = LagiUnpackager::new(&output_path).unpack(None).unwrap();
        assert!(unpacked.get_model().is_some());

        // Roundtrip: fresh graph reads the state back.
        let mem2 = crate::memory::MemoryGraph::new(dir.path().join("lvc2.db"), None).unwrap();
        let imported = load_memory_from_container(&output_path, &mem2).unwrap();
        assert_eq!(imported, 1);
        assert!(mem2.search_text("tokamaks", 1).unwrap().len() == 1);
        assert!(mem2.get_recent(None, 10).unwrap().len() == 1);

        // In-place update replaces the old block (no payload rewrite).
        let entry2 = crate::memory::MemoryEntry::new(
            crate::memory::MemoryType::Context,
            "A second distilled fact for the update test.".to_string(),
            None,
            80,
            vec![],
        );
        memory.store(&entry2).unwrap();
        flush_memory_to_container(&memory, &output_path).unwrap();
        let mem3 = crate::memory::MemoryGraph::new(dir.path().join("lvc3.db"), None).unwrap();
        assert_eq!(load_memory_from_container(&output_path, &mem3).unwrap(), 2);

        // Tampering with the state block does not corrupt payload integrity.
        let mut file = fs::OpenOptions::new().read(true).write(true).open(&output_path).unwrap();
        use std::io::{Seek, Write};
        let len = file.metadata().unwrap().len();
        file.seek(std::io::SeekFrom::Start(len - 8)).unwrap();
        file.write_all(b"TAMPERED").unwrap();
        assert!(LagiUnpackager::new(&output_path).unpack(None).is_ok());
    }
}