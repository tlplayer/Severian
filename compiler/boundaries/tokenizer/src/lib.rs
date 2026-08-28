#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;
use tokenizers::Tokenizer;

pub const ABI_VERSION: u32 = 1;
const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 1;
const PROVIDER_FAILED: i32 = 3;

pub type OpenFn = unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> i32;
pub type EncodeFn =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut i64, *mut u64) -> i32;
pub type ReleaseTokensFn = unsafe extern "C" fn(*mut c_void, *mut i64, u64);
pub type CloseFn = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
pub struct ProviderAbi {
    pub abi_version: u32,
    pub byte_size: u32,
    pub open: OpenFn,
    pub encode: EncodeFn,
    pub release_tokens: ReleaseTokensFn,
    pub close: CloseFn,
}

static PROVIDER: ProviderAbi = ProviderAbi {
    abi_version: ABI_VERSION,
    byte_size: size_of::<ProviderAbi>() as u32,
    open: open,
    encode: encode,
    release_tokens: release_tokens,
    close: close,
};

#[no_mangle]
pub extern "C" fn sev_tokenizer_provider_v1() -> &'static ProviderAbi {
    &PROVIDER
}

unsafe extern "C" fn open(
    _context: *mut c_void,
    path: *const c_char,
    instance: *mut *mut c_void,
) -> i32 {
    if path.is_null() || instance.is_null() {
        return INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The versioned C ABI requires a live NUL-terminated string.
        let path = unsafe { CStr::from_ptr(path) };
        let path = Path::new(path.to_str().map_err(|_| ())?);
        let tokenizer = Tokenizer::from_file(path).map_err(|_| ())?;
        let tokenizer = Box::into_raw(Box::new(tokenizer)).cast::<c_void>();
        // SAFETY: `instance` was validated and is owned by the caller.
        unsafe { instance.write(tokenizer) };
        Ok::<_, ()>(())
    }));
    match result {
        Ok(Ok(())) => OK,
        Ok(Err(())) | Err(_) => PROVIDER_FAILED,
    }
}

unsafe extern "C" fn encode(
    instance: *mut c_void,
    text: *const c_char,
    output: *mut *mut i64,
    count: *mut u64,
) -> i32 {
    if instance.is_null() || text.is_null() || output.is_null() || count.is_null() {
        return INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The handle originates in `open` and remains live until `close`.
        let tokenizer = unsafe { &*instance.cast::<Tokenizer>() };
        // SAFETY: The versioned C ABI requires a live NUL-terminated string.
        let text = unsafe { CStr::from_ptr(text) }.to_str().map_err(|_| ())?;
        let encoded = tokenizer.encode(text, false).map_err(|_| ())?;
        let tokens = encoded
            .get_ids()
            .iter()
            .map(|token| i64::from(*token))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let length = u64::try_from(tokens.len()).map_err(|_| ())?;
        let raw = Box::into_raw(tokens);
        // SAFETY: Both out-pointers were validated and are owned by the caller.
        unsafe {
            output.write((*raw).as_mut_ptr());
            count.write(length);
        }
        Ok::<_, ()>(())
    }));
    match result {
        Ok(Ok(())) => OK,
        Ok(Err(())) | Err(_) => PROVIDER_FAILED,
    }
}

unsafe extern "C" fn release_tokens(_instance: *mut c_void, tokens: *mut i64, count: u64) {
    let Ok(count) = usize::try_from(count) else {
        return;
    };
    if tokens.is_null() && count != 0 {
        return;
    }
    let slice = ptr::slice_from_raw_parts_mut(tokens, count);
    // SAFETY: `encode` allocated this exact boxed slice and transfers it once.
    unsafe { drop(Box::from_raw(slice)) };
}

unsafe extern "C" fn close(instance: *mut c_void) {
    if !instance.is_null() {
        // SAFETY: `open` allocated this tokenizer and transfers it once.
        unsafe { drop(Box::from_raw(instance.cast::<Tokenizer>())) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn canonical_provider_loads_tokenizer_json_and_returns_ids() {
        let model = WordLevel::builder()
            .vocab(ahash::AHashMap::from([
                ("[UNK]".to_owned(), 0),
                ("hello".to_owned(), 7),
            ]))
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        let path = std::env::temp_dir().join(format!(
            "severian-tokenizer-provider-{}.json",
            std::process::id()
        ));
        tokenizer.save(&path, false).unwrap();
        let path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let text = CString::new("hello").unwrap();
        let mut instance = ptr::null_mut();
        let mut tokens = ptr::null_mut();
        let mut count = 0;
        unsafe {
            assert_eq!(open(ptr::null_mut(), path.as_ptr(), &mut instance), OK);
            assert_eq!(encode(instance, text.as_ptr(), &mut tokens, &mut count), OK);
            assert_eq!(std::slice::from_raw_parts(tokens, count as usize), [7]);
            release_tokens(instance, tokens, count);
            close(instance);
        }
        let _ = std::fs::remove_file(path.to_str().unwrap());
    }

    #[test]
    fn pinned_omnivoice_tokenizer_matches_known_prompt_ids_when_installed() {
        let Some(path) = std::env::var_os("SEVERIAN_OMNIVOICE_TOKENIZER") else {
            return;
        };
        let path = CString::new(path.as_encoded_bytes()).unwrap();
        let text =
            CString::new("<|text_start|>The quick brown fox jumps over the lazy dog.<|text_end|>")
                .unwrap();
        let mut instance = ptr::null_mut();
        let mut tokens = ptr::null_mut();
        let mut count = 0;
        unsafe {
            assert_eq!(open(ptr::null_mut(), path.as_ptr(), &mut instance), OK);
            assert_eq!(encode(instance, text.as_ptr(), &mut tokens, &mut count), OK);
            assert_eq!(
                std::slice::from_raw_parts(tokens, count as usize),
                [151674, 785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 151675]
            );
            release_tokens(instance, tokens, count);
            close(instance);
        }
    }
}
