/*!
Binary Specification Constants & Struct Definitions for .lagi Format
*/

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};
use thiserror::Error;

/// Magic bytes identifying the .lagi format
pub const MAGIC_BYTES: &[u8; 8] = b"LAGI_V01";

/// Current format version
pub const FORMAT_VERSION: u16 = 1;

/// Header format size: 8 + 2 + 2 + 4 + 4 + 12 + 32 = 64 bytes
pub const HEADER_SIZE: usize = 64;

/// Header format string for documentation: "!8sHHII12s32s"
pub const HEADER_FORMAT: &str = "8s HH II 12s 32s";

/// Maximum manifest size (1 MB)
pub const MAX_MANIFEST_SIZE: usize = 1024 * 1024;

/// Maximum payload size (4 GB)
pub const MAX_PAYLOAD_SIZE: usize = 4 * 1024 * 1024 * 1024;


/// Encryption mode identifiers for the .lagi container
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u16)]
pub enum EncryptionMode {
    Plaintext = 0,
    Aes256GcmHardware = 1,
    Aes256GcmEnclave = 2,
}

impl EncryptionMode {
    pub fn from_u16(val: u16) -> Option<Self> {
        match val {
            0 => Some(EncryptionMode::Plaintext),
            1 => Some(EncryptionMode::Aes256GcmHardware),
            2 => Some(EncryptionMode::Aes256GcmEnclave),
            _ => None,
        }
    }

    pub fn to_u16(self) -> u16 {
        self as u16
    }
}

/// Header flags (reserved for future use)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Flags {
    None = 0,
    Compressed = 1 << 0,
    Signed = 1 << 1,
}

impl Flags {
    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => Flags::None,
            1 => Flags::Compressed,
            2 => Flags::Signed,
            _ => Flags::None,
        }
    }

    pub fn to_u32(self) -> u32 {
        self as u32
    }
}

/// Binary header structure
#[derive(Debug, Clone)]
pub struct LagiHeader {
    pub magic: [u8; 8],
    pub version: u16,
    pub enc_mode: EncryptionMode,
    pub manifest_len: u32,
    pub reserved: u32,
    pub nonce: [u8; 12],
    pub payload_hash: [u8; 32],
}

impl LagiHeader {
    /// Pack header into bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.write_all(&self.magic).unwrap();
        buf.write_u16::<BigEndian>(self.version).unwrap();
        buf.write_u16::<BigEndian>(self.enc_mode.to_u16()).unwrap();
        buf.write_u32::<BigEndian>(self.manifest_len).unwrap();
        buf.write_u32::<BigEndian>(self.reserved).unwrap();
        buf.write_all(&self.nonce).unwrap();
        buf.write_all(&self.payload_hash).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);
        buf
    }

    /// Unpack header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, SpecError> {
        if data.len() != HEADER_SIZE {
            return Err(SpecError::InvalidHeaderSize { expected: HEADER_SIZE, got: data.len() });
        }

        let mut cursor = Cursor::new(data);
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        
        let version = cursor.read_u16::<BigEndian>()?;
        let enc_mode_raw = cursor.read_u16::<BigEndian>()?;
        let enc_mode = EncryptionMode::from_u16(enc_mode_raw)
            .ok_or(SpecError::InvalidEncryptionMode(enc_mode_raw))?;
        
        let manifest_len = cursor.read_u32::<BigEndian>()?;
        let reserved = cursor.read_u32::<BigEndian>()?;
        
        let mut nonce = [0u8; 12];
        cursor.read_exact(&mut nonce)?;
        
        let mut payload_hash = [0u8; 32];
        cursor.read_exact(&mut payload_hash)?;

        Ok(LagiHeader {
            magic,
            version,
            enc_mode,
            manifest_len,
            reserved,
            nonce,
            payload_hash,
        })
    }

    /// Verify magic bytes
    pub fn verify_magic(&self) -> bool {
        self.magic == *MAGIC_BYTES
    }
}

/// Pack header from components
pub fn pack_header(
    magic: &[u8; 8],
    version: u16,
    enc_mode: EncryptionMode,
    manifest_len: u32,
    reserved: u32,
    nonce: &[u8; 12],
    payload_hash: &[u8; 32],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_SIZE);
    buf.write_all(magic).unwrap();
    buf.write_u16::<BigEndian>(version).unwrap();
    buf.write_u16::<BigEndian>(enc_mode.to_u16()).unwrap();
    buf.write_u32::<BigEndian>(manifest_len).unwrap();
    buf.write_u32::<BigEndian>(reserved).unwrap();
    buf.write_all(nonce).unwrap();
    buf.write_all(payload_hash).unwrap();
    assert_eq!(buf.len(), HEADER_SIZE);
    buf
}

/// Unpack header from bytes
pub fn unpack_header(data: &[u8]) -> Result<(Vec<u8>, u16, u16, u32, u32, Vec<u8>, Vec<u8>), SpecError> {
    if data.len() != HEADER_SIZE {
        return Err(SpecError::InvalidHeaderSize { expected: HEADER_SIZE, got: data.len() });
    }

    let mut cursor = Cursor::new(data);
    let mut magic = vec![0u8; 8];
    cursor.read_exact(&mut magic)?;
    
    let version = cursor.read_u16::<BigEndian>()?;
    let enc_mode = cursor.read_u16::<BigEndian>()?;
    let manifest_len = cursor.read_u32::<BigEndian>()?;
    let reserved = cursor.read_u32::<BigEndian>()?;
    
    let mut nonce = vec![0u8; 12];
    cursor.read_exact(&mut nonce)?;
    
    let mut payload_hash = vec![0u8; 32];
    cursor.read_exact(&mut payload_hash)?;

    Ok((magic, version, enc_mode, manifest_len, reserved, nonce, payload_hash))
}

/// Verify magic bytes
pub fn verify_magic(magic: &[u8]) -> bool {
    magic == MAGIC_BYTES
}

/// Specification errors
#[derive(Error, Debug)]
pub enum SpecError {
    #[error("Invalid header size: expected {expected}, got {got}")]
    InvalidHeaderSize { expected: usize, got: usize },
    
    #[error("Invalid magic bytes")]
    InvalidMagic,
    
    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u16),
    
    #[error("Invalid encryption mode: {0}")]
    InvalidEncryptionMode(u16),
    
    #[error("Manifest too large: {0} bytes (max {})", MAX_MANIFEST_SIZE)]
    ManifestTooLarge(u32),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = LagiHeader {
            magic: *MAGIC_BYTES,
            version: FORMAT_VERSION,
            enc_mode: EncryptionMode::Aes256GcmHardware,
            manifest_len: 100,
            reserved: 0,
            nonce: [1u8; 12],
            payload_hash: [2u8; 32],
        };
        
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        
        let parsed = LagiHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.magic, header.magic);
        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.enc_mode, header.enc_mode);
        assert_eq!(parsed.manifest_len, header.manifest_len);
        assert_eq!(parsed.nonce, header.nonce);
        assert_eq!(parsed.payload_hash, header.payload_hash);
    }

    #[test]
    fn test_verify_magic() {
        let header = LagiHeader {
            magic: *MAGIC_BYTES,
            version: FORMAT_VERSION,
            enc_mode: EncryptionMode::Aes256GcmHardware,
            manifest_len: 100,
            reserved: 0,
            nonce: [0u8; 12],
            payload_hash: [0u8; 32],
        };
        assert!(header.verify_magic());
        
        let mut bad_header = header;
        bad_header.magic = *b"BAD_MAGI";
        assert!(!bad_header.verify_magic());
    }

    #[test]
    fn test_pack_unpack_header() {
        let magic = *MAGIC_BYTES;
        let version = FORMAT_VERSION;
        let enc_mode = EncryptionMode::Aes256GcmHardware;
        let manifest_len = 256;
        let reserved = 0;
        let nonce = [42u8; 12];
        let payload_hash = [99u8; 32];
        
        let packed = pack_header(&magic, version, enc_mode, manifest_len, reserved, &nonce, &payload_hash);
        let (m, v, e, ml, r, n, ph) = unpack_header(&packed).unwrap();
        
        assert_eq!(m, magic.to_vec());
        assert_eq!(v, version);
        assert_eq!(e, enc_mode.to_u16());
        assert_eq!(ml, manifest_len);
        assert_eq!(r, reserved);
        assert_eq!(n, nonce.to_vec());
        assert_eq!(ph, payload_hash.to_vec());
    }
}