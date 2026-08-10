use crate::{
    cache::CacheKey,
    pjrt::{
        api,
        compile::RawExecutable,
        error,
        platform::borrowed_string,
    },
    Result,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompiledMemoryStats {
    pub generated_code_size: i64,
    pub argument_size: i64,
    pub output_size: i64,
    pub alias_size: i64,
    pub temporary_size: i64,
    pub host_generated_code_size: i64,
    pub host_argument_size: i64,
    pub host_output_size: i64,
    pub host_alias_size: i64,
    pub host_temporary_size: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutableMetadata {
    pub name: String,
    pub replicas: usize,
    pub partitions: usize,
    pub outputs: usize,
    pub generated_code_size: i64,
    pub memory: Option<CompiledMemoryStats>,
    pub fingerprint: Option<Vec<u8>>,
}

impl RawExecutable {
    pub fn metadata(&self) -> Result<ExecutableMetadata> {
        Ok(ExecutableMetadata {
            name: self.name()?,
            replicas: self.num_replicas()?,
            partitions: self.num_partitions()?,
            outputs: self.num_outputs()?,
            generated_code_size: self.generated_code_size()?,
            memory: self.compiled_memory_stats().ok(),
            fingerprint: self.fingerprint().ok().map(|value| value.0),
        })
    }

    pub fn name(&self) -> Result<String> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Executable_Name_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_Name_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
            executable_name: std::ptr::null(),
            executable_name_size: 0,
        };
        let result = unsafe { (api.PJRT_Executable_Name)(&mut args) };
        unsafe { error::check(api, result)? };
        borrowed_string(args.executable_name, args.executable_name_size)
    }

    pub fn num_replicas(&self) -> Result<usize> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Executable_NumReplicas_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_NumReplicas_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
            num_replicas: 0,
        };
        let result = unsafe { (api.PJRT_Executable_NumReplicas)(&mut args) };
        unsafe { error::check(api, result)? };
        Ok(args.num_replicas)
    }

    pub fn num_partitions(&self) -> Result<usize> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Executable_NumPartitions_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_NumPartitions_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
            num_partitions: 0,
        };
        let result = unsafe { (api.PJRT_Executable_NumPartitions)(&mut args) };
        unsafe { error::check(api, result)? };
        Ok(args.num_partitions)
    }

    pub fn generated_code_size(&self) -> Result<i64> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
            size_in_bytes: 0,
        };
        let result = unsafe { (api.PJRT_Executable_SizeOfGeneratedCodeInBytes)(&mut args) };
        unsafe { error::check(api, result)? };
        Ok(args.size_in_bytes)
    }

    pub fn compiled_memory_stats(&self) -> Result<CompiledMemoryStats> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Executable_GetCompiledMemoryStats_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_GetCompiledMemoryStats_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
            generated_code_size_in_bytes: 0,
            argument_size_in_bytes: 0,
            output_size_in_bytes: 0,
            alias_size_in_bytes: 0,
            temp_size_in_bytes: 0,
            host_generated_code_size_in_bytes: 0,
            host_argument_size_in_bytes: 0,
            host_output_size_in_bytes: 0,
            host_alias_size_in_bytes: 0,
            host_temp_size_in_bytes: 0,
        };
        let result = unsafe { (api.PJRT_Executable_GetCompiledMemoryStats)(&mut args) };
        unsafe { error::check(api, result)? };

        Ok(CompiledMemoryStats {
            generated_code_size: args.generated_code_size_in_bytes,
            argument_size: args.argument_size_in_bytes,
            output_size: args.output_size_in_bytes,
            alias_size: args.alias_size_in_bytes,
            temporary_size: args.temp_size_in_bytes,
            host_generated_code_size: args.host_generated_code_size_in_bytes,
            host_argument_size: args.host_argument_size_in_bytes,
            host_output_size: args.host_output_size_in_bytes,
            host_alias_size: args.host_alias_size_in_bytes,
            host_temporary_size: args.host_temp_size_in_bytes,
        })
    }
}

/// Human-readable cache manifest.
///
/// It is deliberately trivial to inspect and recover if the schema grows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutableManifest {
    pub format_version: u32,
    pub cache_key: CacheKey,
    pub platform_name: String,
    pub platform_version: String,
    pub pjrt_api_major: i32,
    pub pjrt_api_minor: i32,
    pub executable_fingerprint: Option<String>,
    pub generated_code_size: Option<i64>,
}

impl ExecutableManifest {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn encode(&self) -> String {
        let mut lines = vec![
            format!("format_version={}", self.format_version),
            format!("cache_key={}", self.cache_key),
            format!("platform_name={}", escape(&self.platform_name)),
            format!("platform_version={}", escape(&self.platform_version)),
            format!("pjrt_api_major={}", self.pjrt_api_major),
            format!("pjrt_api_minor={}", self.pjrt_api_minor),
        ];

        if let Some(fingerprint) = &self.executable_fingerprint {
            lines.push(format!(
                "executable_fingerprint={}",
                escape(fingerprint)
            ));
        }

        if let Some(bytes) = self.generated_code_size {
            lines.push(format!("generated_code_size={bytes}"));
        }

        lines.push(String::new());
        lines.join("\n")
    }

    pub fn decode(text: &str) -> Result<Self, String> {
        let mut values = std::collections::HashMap::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("invalid manifest line `{line}`"))?;
            values.insert(key.to_owned(), unescape(value)?);
        }

        let format_version = parse_required(&values, "format_version")?;
        let cache_key = parse_cache_key(required(&values, "cache_key")?)?;
        let platform_name = required(&values, "platform_name")?.to_owned();
        let platform_version = required(&values, "platform_version")?.to_owned();
        let pjrt_api_major = parse_required(&values, "pjrt_api_major")?;
        let pjrt_api_minor = parse_required(&values, "pjrt_api_minor")?;
        let executable_fingerprint = values.get("executable_fingerprint").cloned();
        let generated_code_size = values
            .get("generated_code_size")
            .map(|value| value.parse::<i64>())
            .transpose()
            .map_err(|error| error.to_string())?;

        Ok(Self {
            format_version,
            cache_key,
            platform_name,
            platform_version,
            pjrt_api_major,
            pjrt_api_minor,
            executable_fingerprint,
            generated_code_size,
        })
    }
}

fn required<'a>(
    values: &'a std::collections::HashMap<String, String>,
    key: &str,
) -> Result<&'a str, String> {
    values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("manifest missing `{key}`"))
}

fn parse_required<T: std::str::FromStr>(
    values: &std::collections::HashMap<String, String>,
    key: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    required(values, key)?
        .parse()
        .map_err(|error| format!("invalid `{key}`: {error}"))
}

fn parse_cache_key(value: &str) -> Result<CacheKey, String> {
    if value.len() != 32 {
        return Err("cache key must contain 32 hexadecimal characters".into());
    }

    let high = u64::from_str_radix(&value[..16], 16)
        .map_err(|error| error.to_string())?;
    let low = u64::from_str_radix(&value[16..], 16)
        .map_err(|error| error.to_string())?;

    Ok(CacheKey::new(high, low))
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('=', "\\=")
}

fn unescape(value: &str) -> Result<String, String> {
    let mut output = String::new();
    let mut chars = value.chars();

    while let Some(character) = chars.next() {
        if character != '\\' {
            output.push(character);
            continue;
        }

        match chars.next() {
            Some('n') => output.push('\n'),
            Some('\\') => output.push('\\'),
            Some('=') => output.push('='),
            Some(other) => {
                output.push('\\');
                output.push(other);
            }
            None => return Err("manifest ends with an incomplete escape".into()),
        }
    }

    Ok(output)
}
