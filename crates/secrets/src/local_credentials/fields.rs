/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::file::FileWatcher;
use crate::SecretsError;
use crate::credentials::{CredentialReader, Credentials};

/// Source for one credential field. Inline values are used verbatim; files
/// contain UTF-8 text with one trailing LF or CRLF removed, preserving spaces.
/// Relative paths resolve from the process working directory.
#[derive(Clone)]
pub enum CredentialValue {
    /// A value supplied directly by the caller.
    Inline(String),
    /// A file whose contents supply the value.
    File(PathBuf),
}

impl fmt::Debug for CredentialValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline(_) => f.debug_tuple("Inline").field(&"REDACTED").finish(),
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
        }
    }
}

/// Independent sources for a username/password pair, allowing either field
/// to remain inline while the other comes from a watched file.
#[derive(Clone, Debug)]
pub struct CredentialSources {
    /// Source for the username.
    pub username: CredentialValue,
    /// Source for the password.
    pub password: CredentialValue,
}

/// Cached, read-only credentials indexed by caller-defined keys, such as BMC
/// MAC addresses. This reader does not require a Vault client or writer.
///
/// One kernel watcher and one polling watcher observe all distinct configured
/// paths, regardless of the number of keys sharing those paths. File events
/// trigger reloads; polling also detects replacements and symlink updates.
/// If the kernel watcher cannot start, observation falls back to polling.
/// Lookups only read memory. A failed reload retains the entire last valid
/// snapshot, and dropping the reader stops observation.
pub struct FileCredentialReader<K> {
    watcher: FileWatcher<HashMap<K, Credentials>>,
}

impl<K: Clone + Eq + Hash + Send + Sync + 'static> FileCredentialReader<K> {
    /// Loads all credential pairs before returning and starts file observation.
    /// Missing or invalid UTF-8 files fail construction. `poll_interval` must
    /// be positive; 60 seconds matches the existing structured-file reader.
    /// An unknown key returns `None`, allowing use in a reader chain.
    ///
    /// ```no_run
    /// use std::collections::HashMap;
    /// use std::time::Duration;
    ///
    /// use carbide_secrets::credentials::CredentialReader;
    /// use carbide_secrets::local_credentials::{
    ///     CredentialSources, CredentialValue, FileCredentialReader,
    /// };
    ///
    /// # async fn example() -> Result<(), carbide_secrets::SecretsError> {
    /// let bmc = mac_address::MacAddress::new([0, 1, 2, 3, 4, 5]);
    /// let reader = FileCredentialReader::new(
    ///     HashMap::from([(
    ///         bmc,
    ///         CredentialSources {
    ///             username: CredentialValue::Inline("root".into()),
    ///             password: CredentialValue::File("/run/secrets/bmc/password".into()),
    ///         },
    ///     )]),
    ///     Duration::from_secs(60),
    /// )
    /// .await?;
    /// let credentials = reader.get_credentials(&bmc).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(
        sources: HashMap<K, CredentialSources>,
        poll_interval: Duration,
    ) -> Result<Self, SecretsError> {
        let paths: HashSet<PathBuf> = sources
            .values()
            .flat_map(|source| {
                [&source.username, &source.password]
                    .into_iter()
                    .filter_map(|value| match value {
                        CredentialValue::File(path) => Some(path.clone()),
                        _ => None,
                    })
            })
            .collect();
        let sources = Arc::new(sources);
        let watcher = FileWatcher::new(paths.into_iter().collect(), poll_interval, move || {
            let sources = sources.clone();
            async move { Self::load(&sources).await }
        })
        .await?;
        Ok(Self { watcher })
    }

    async fn load(
        sources: &HashMap<K, CredentialSources>,
    ) -> Result<HashMap<K, Credentials>, SecretsError> {
        // A shared field is read only once per reload, even when many BMCs use it.
        let mut files = HashMap::new();
        let mut credentials = HashMap::with_capacity(sources.len());
        for (key, source) in sources {
            let username = source.username.read(&mut files).await?;
            let password = source.password.read(&mut files).await?;
            credentials.insert(key.clone(), Credentials::new(username, password));
        }
        Ok(credentials)
    }
}

impl CredentialValue {
    async fn read(&self, files: &mut HashMap<PathBuf, String>) -> Result<String, SecretsError> {
        match self {
            Self::Inline(text) => Ok(text.clone()),
            Self::File(path) => {
                if !files.contains_key(path) {
                    let text =
                        tokio::fs::read_to_string(path).await.map_err(|err| {
                            SecretsError::GenericError(eyre::Report::new(err).wrap_err(format!(
                                "failed to read credential file {}",
                                path.display()
                            )))
                        })?;
                    let text = text
                        .strip_suffix("\r\n")
                        .or_else(|| text.strip_suffix('\n'))
                        .unwrap_or(&text)
                        .to_owned();
                    files.insert(path.clone(), text);
                }
                Ok(files[path].clone())
            }
        }
    }
}

#[async_trait]
impl<K: Clone + Eq + Hash + Send + Sync + 'static> CredentialReader<K> for FileCredentialReader<K> {
    async fn get_credentials(&self, key: &K) -> Result<Option<Credentials>, SecretsError> {
        Ok(self.watcher.credentials.load().get(key).cloned())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tempfile::tempdir;

    use super::*;
    use crate::ChainedCredentialReader;

    #[tokio::test]
    async fn field_values_preserve_spaces_and_remove_one_line_ending() {
        let dir = tempdir().expect("temporary directory");
        let path = dir.path().join("value");
        for (input, expected) in [
            ("  secret  ", "  secret  "),
            ("secret\n", "secret"),
            ("secret\r\n", "secret"),
            ("secret\n\n", "secret\n"),
        ] {
            tokio::fs::write(&path, input).await.expect("write value");
            let actual = CredentialValue::File(path.clone())
                .read(&mut HashMap::new())
                .await
                .expect("read value");
            assert_eq!(actual, expected, "input {input:?}");
        }
        assert_eq!(
            CredentialValue::Inline("inline\n".into())
                .read(&mut HashMap::new())
                .await
                .expect("inline value"),
            "inline\n"
        );
    }

    #[tokio::test]
    async fn per_target_sources_work_in_a_read_only_chain() {
        let dir = tempdir().expect("temporary directory");
        let username = dir.path().join("username");
        let password = dir.path().join("password");
        tokio::fs::write(&username, "root\n")
            .await
            .expect("write username");
        tokio::fs::write(&password, "shared\n")
            .await
            .expect("write password");
        let reader = Arc::new(
            FileCredentialReader::new(
                HashMap::from([
                    (
                        1,
                        CredentialSources {
                            username: CredentialValue::File(username),
                            password: CredentialValue::File(password.clone()),
                        },
                    ),
                    (
                        2,
                        CredentialSources {
                            username: CredentialValue::Inline("admin".into()),
                            password: CredentialValue::File(password.clone()),
                        },
                    ),
                ]),
                Duration::from_millis(50),
            )
            .await
            .expect("file reader"),
        );
        let fallback = FileCredentialReader::new(
            HashMap::from([(
                3,
                CredentialSources {
                    username: CredentialValue::Inline("fallback".into()),
                    password: CredentialValue::Inline("local".into()),
                },
            )]),
            Duration::from_secs(60),
        )
        .await
        .expect("inline reader");
        let chain: ChainedCredentialReader<i32> = vec![
            Box::new(reader.clone()) as Box<dyn CredentialReader<i32>>,
            Box::new(fallback),
        ]
        .into();
        for (key, expected) in [
            (1, Some(Credentials::new("root", "shared"))),
            (2, Some(Credentials::new("admin", "shared"))),
            (3, Some(Credentials::new("fallback", "local"))),
            (4, None),
        ] {
            assert_eq!(
                chain.get_credentials(&key).await.expect("chain lookup"),
                expected,
                "key {key}"
            );
        }
        let replacement = dir.path().join("next");
        tokio::fs::write(&replacement, "rotated")
            .await
            .expect("write replacement");
        tokio::fs::rename(replacement, &password)
            .await
            .expect("replace password");
        wait_for(&reader, &1, Credentials::new("root", "rotated")).await;
        assert_eq!(
            reader
                .get_credentials(&2)
                .await
                .expect("shared source lookup"),
            Some(Credentials::new("admin", "rotated"))
        );
    }

    #[tokio::test]
    async fn failed_reload_keeps_cached_credentials_and_recovers() {
        let dir = tempdir().expect("temporary directory");
        let path = dir.path().join("password");
        tokio::fs::write(&path, "before")
            .await
            .expect("write password");
        let sources = Arc::new(HashMap::from([(
            1,
            CredentialSources {
                username: CredentialValue::Inline("root".into()),
                password: CredentialValue::File(path.clone()),
            },
        )]));
        let failures = Arc::new(AtomicUsize::new(0));
        let failures_clone = failures.clone();
        let watcher = FileWatcher::new(vec![path.clone()], Duration::from_millis(50), move || {
            let sources = sources.clone();
            let failures = failures_clone.clone();
            async move {
                let result = FileCredentialReader::load(&sources).await;
                if result.is_err() {
                    failures.fetch_add(1, Ordering::SeqCst);
                }
                result
            }
        })
        .await
        .expect("initial load");
        let reader = FileCredentialReader { watcher };
        tokio::fs::write(&path, [0xff])
            .await
            .expect("write invalid UTF-8");
        tokio::time::timeout(Duration::from_secs(3), async {
            while failures.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("observe actual failed reread");
        assert_eq!(
            reader
                .get_credentials(&1)
                .await
                .expect("cached lookup after failed reread"),
            Some(Credentials::new("root", "before"))
        );
        tokio::fs::write(&path, "after")
            .await
            .expect("restore file");
        wait_for(&reader, &1, Credentials::new("root", "after")).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn follows_projected_secret_symlink_updates() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().expect("temporary directory");
        for (revision, password) in [("v1", "before"), ("v2", "after")] {
            let revision_dir = dir.path().join(revision);
            std::fs::create_dir(&revision_dir).expect("revision directory");
            std::fs::write(revision_dir.join("password"), password).expect("revision password");
        }
        symlink("v1", dir.path().join("..data")).expect("active revision");
        symlink("..data/password", dir.path().join("password")).expect("projected key");
        let reader = FileCredentialReader::new(
            HashMap::from([(
                1,
                CredentialSources {
                    username: CredentialValue::Inline("root".into()),
                    password: CredentialValue::File(dir.path().join("password")),
                },
            )]),
            Duration::from_millis(50),
        )
        .await
        .expect("reader");
        symlink("v2", dir.path().join("..data_tmp")).expect("new revision");
        std::fs::rename(dir.path().join("..data_tmp"), dir.path().join("..data"))
            .expect("rotate Secret");
        wait_for(&reader, &1, Credentials::new("root", "after")).await;
    }

    #[tokio::test]
    async fn invalid_sources_fail_initial_load() {
        let dir = tempdir().expect("temporary directory");
        let missing = dir.path().join("missing");
        let invalid = dir.path().join("invalid");
        tokio::fs::write(&invalid, [0xff])
            .await
            .expect("invalid UTF-8");
        for (scenario, source, interval) in [
            (
                "missing file",
                CredentialValue::File(missing),
                Duration::from_secs(60),
            ),
            (
                "invalid UTF-8",
                CredentialValue::File(invalid),
                Duration::from_secs(60),
            ),
            (
                "zero interval",
                CredentialValue::Inline("password".into()),
                Duration::ZERO,
            ),
        ] {
            let result = FileCredentialReader::new(
                HashMap::from([(
                    1,
                    CredentialSources {
                        username: CredentialValue::Inline("root".into()),
                        password: source,
                    },
                )]),
                interval,
            )
            .await;
            assert!(result.is_err(), "{scenario}");
        }
    }

    #[test]
    fn inline_debug_redacts_values() {
        assert_eq!(
            format!("{:?}", CredentialValue::Inline("sensitive".into())),
            "Inline(\"REDACTED\")",
        );
    }

    async fn wait_for(reader: &FileCredentialReader<i32>, key: &i32, expected: Credentials) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if reader.get_credentials(key).await.expect("cached lookup")
                    == Some(expected.clone())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("credentials reload");
    }
}
