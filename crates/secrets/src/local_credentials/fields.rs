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

use super::UsernamePassword;
use super::file::{FileWatcher, report_reload_error};
use crate::SecretsError;
use crate::credentials::CredentialReader;

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
    /// Source for the password. `None` preserves an absent password; an empty
    /// inline value or empty file produces `Some(String::new())`.
    pub password: Option<CredentialValue>,
}

/// Cached, read-only credentials indexed by caller-defined keys, such as BMC
/// MAC addresses. This reader does not require a Vault client or writer.
///
/// One kernel watcher and one polling watcher observe all distinct configured
/// paths, regardless of the number of keys sharing those paths. File events
/// trigger reloads. Symlink updates, including Kubernetes Secret projections,
/// are detected within `poll_interval`, plus time to read the files.
/// If the kernel watcher cannot start, observation falls back to polling.
/// Lookups only read memory. A failed read retains the last valid pair for that
/// key, while other keys continue updating. A configured key without a valid
/// initial pair returns an error until it recovers; an unknown key returns
/// `None`. Dropping the reader stops observation.
pub struct FileCredentialReader<K> {
    watcher: FileWatcher<CachedCredentials<K>>,
}

impl<K: Clone + Eq + Hash + Send + Sync + 'static> FileCredentialReader<K> {
    /// Attempts each credential pair before returning and starts observation.
    /// Missing or invalid UTF-8 files affect only their configured keys.
    /// Failed files are retried every `poll_interval`. `poll_interval` must
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
    ///             password: Some(CredentialValue::File("/run/secrets/bmc/password".into())),
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
                std::iter::once(&source.username)
                    .chain(source.password.as_ref())
                    .filter_map(|value| match value {
                        CredentialValue::File(path) => Some(path.clone()),
                        _ => None,
                    })
            })
            .collect();
        let sources = Arc::new(sources);
        let watcher = FileWatcher::new(
            paths.into_iter().collect(),
            poll_interval,
            move |previous| {
                let sources = sources.clone();
                async move { Ok(Self::load(&sources, previous).await) }
            },
        )
        .await?;
        Ok(Self { watcher })
    }

    async fn load(
        sources: &HashMap<K, CredentialSources>,
        previous: Option<Arc<CachedCredentials<K>>>,
    ) -> CachedCredentials<K> {
        // A shared field is read only once per reload, including failed reads.
        let mut files = HashMap::new();
        let mut credentials = previous.as_deref().cloned().unwrap_or_default();
        for (key, source) in sources {
            match source.read(&mut files).await {
                Ok(updated) => {
                    credentials.insert(key.clone(), Ok(updated));
                }
                Err(error) => {
                    if !matches!(credentials.get(key), Some(Ok(_))) {
                        credentials.insert(key.clone(), Err(error));
                    }
                }
            }
        }
        credentials
    }
}

type CachedCredentials<K> =
    HashMap<K, Result<UsernamePassword<Option<String>>, Arc<CredentialFileError>>>;
type FileReadCache = HashMap<PathBuf, Result<String, Arc<CredentialFileError>>>;

#[derive(Debug, thiserror::Error)]
#[error("failed to read credential file {path}: {source}")]
struct CredentialFileError {
    path: PathBuf,
    #[source]
    source: std::io::Error,
}

impl CredentialSources {
    async fn read(
        &self,
        files: &mut FileReadCache,
    ) -> Result<UsernamePassword<Option<String>>, Arc<CredentialFileError>> {
        let username = self.username.read(files).await?;
        let password = match &self.password {
            Some(source) => Some(source.read(files).await?),
            None => None,
        };
        Ok(UsernamePassword { username, password })
    }
}

impl CredentialValue {
    async fn read(&self, files: &mut FileReadCache) -> Result<String, Arc<CredentialFileError>> {
        match self {
            Self::Inline(text) => Ok(text.clone()),
            Self::File(path) => {
                if !files.contains_key(path) {
                    let result = tokio::fs::read_to_string(path)
                        .await
                        .map(|text| {
                            text.strip_suffix("\r\n")
                                .or_else(|| text.strip_suffix('\n'))
                                .unwrap_or(&text)
                                .to_owned()
                        })
                        .map_err(|source| {
                            let error = Arc::new(CredentialFileError {
                                path: path.clone(),
                                source,
                            });
                            report_reload_error(&error);
                            error
                        });
                    files.insert(path.clone(), result);
                }
                files[path].clone()
            }
        }
    }
}

#[async_trait]
impl<K: Clone + Eq + Hash + Send + Sync + 'static>
    CredentialReader<K, UsernamePassword<Option<String>>> for FileCredentialReader<K>
{
    async fn get_credentials(
        &self,
        key: &K,
    ) -> Result<Option<UsernamePassword<Option<String>>>, SecretsError> {
        match self.watcher.credentials.load().get(key) {
            Some(Ok(credentials)) => Ok(Some(credentials.clone())),
            Some(Err(error)) => Err(SecretsError::GenericError(eyre::Report::new(error.clone()))),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
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
                            password: Some(CredentialValue::File(password.clone())),
                        },
                    ),
                    (
                        2,
                        CredentialSources {
                            username: CredentialValue::Inline("admin".into()),
                            password: Some(CredentialValue::File(password.clone())),
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
                    password: Some(CredentialValue::Inline("local".into())),
                },
            )]),
            Duration::from_secs(60),
        )
        .await
        .expect("inline reader");
        let chain: ChainedCredentialReader<i32, UsernamePassword<Option<String>>> = vec![
            Box::new(reader.clone())
                as Box<dyn CredentialReader<i32, UsernamePassword<Option<String>>>>,
            Box::new(fallback),
        ]
        .into();
        for (key, expected) in [
            (1, Some(pair("root", "shared"))),
            (2, Some(pair("admin", "shared"))),
            (3, Some(pair("fallback", "local"))),
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
        wait_for(&reader, &1, pair("root", "rotated")).await;
        assert_eq!(
            reader
                .get_credentials(&2)
                .await
                .expect("shared source lookup"),
            Some(pair("admin", "rotated"))
        );
    }

    #[tokio::test]
    #[serial_test::serial(credential_watcher_metrics)]
    async fn failed_key_keeps_its_pair_while_other_keys_rotate() {
        let dir = tempdir().expect("temporary directory");
        let username = dir.path().join("username");
        let broken_password = dir.path().join("broken-password");
        let healthy_password = dir.path().join("healthy-password");
        for (path, value) in [
            (&username, "before-user"),
            (&broken_password, "before"),
            (&healthy_password, "before"),
        ] {
            tokio::fs::write(path, value)
                .await
                .expect("initial credential field");
        }
        let reader = FileCredentialReader::new(
            HashMap::from([
                (
                    1,
                    CredentialSources {
                        username: CredentialValue::File(username.clone()),
                        password: Some(CredentialValue::File(broken_password.clone())),
                    },
                ),
                (
                    2,
                    CredentialSources {
                        username: CredentialValue::Inline("healthy".into()),
                        password: Some(CredentialValue::File(healthy_password.clone())),
                    },
                ),
            ]),
            Duration::from_millis(50),
        )
        .await
        .expect("initial reader");
        tokio::fs::write(&broken_password, [0xff])
            .await
            .expect("break one password");
        tokio::fs::write(&username, "after-user")
            .await
            .expect("rotate failed key username");
        tokio::fs::write(&healthy_password, "after")
            .await
            .expect("rotate healthy key");
        wait_for(&reader, &2, pair("healthy", "after")).await;
        assert_eq!(
            reader.get_credentials(&1).await.expect("last valid pair"),
            Some(pair("before-user", "before"))
        );
        tokio::fs::write(&broken_password, "after")
            .await
            .expect("repair broken key");
        wait_for(&reader, &1, pair("after-user", "after")).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn follows_projected_secret_symlink_updates() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().expect("temporary directory");
        for (revision, password) in [("v1", "before"), ("v2", "after"), ("v3", "again")] {
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
                    password: Some(CredentialValue::File(dir.path().join("password"))),
                },
            )]),
            Duration::from_millis(50),
        )
        .await
        .expect("reader");
        for (revision, password) in [("v2", "after"), ("v3", "again")] {
            symlink(revision, dir.path().join("..data_tmp")).expect("new revision");
            std::fs::rename(dir.path().join("..data_tmp"), dir.path().join("..data"))
                .expect("rotate Secret");
            wait_for(&reader, &1, pair("root", password)).await;
        }
    }

    #[tokio::test]
    #[serial_test::serial(credential_watcher_metrics)]
    async fn invalid_initial_keys_do_not_block_healthy_keys_and_recover() {
        for scenario in ["missing file", "invalid UTF-8"] {
            let dir = tempdir().expect("temporary directory");
            let path = dir.path().join("password");
            if scenario == "invalid UTF-8" {
                tokio::fs::write(&path, [0xff])
                    .await
                    .expect("invalid UTF-8");
            }
            let reader = FileCredentialReader::new(
                HashMap::from([
                    (
                        1,
                        CredentialSources {
                            username: CredentialValue::Inline("broken".into()),
                            password: Some(CredentialValue::File(path.clone())),
                        },
                    ),
                    (
                        2,
                        CredentialSources {
                            username: CredentialValue::Inline("healthy".into()),
                            password: Some(CredentialValue::Inline("valid".into())),
                        },
                    ),
                ]),
                Duration::from_millis(50),
            )
            .await
            .expect("construction survives one bad key");
            let error = reader.get_credentials(&1).await.expect_err(scenario);
            let report: eyre::Report = error.into();
            assert!(
                report
                    .chain()
                    .any(|cause| cause.downcast_ref::<std::io::Error>().is_some()),
                "{scenario}: preserve the I/O error"
            );
            assert_eq!(
                reader
                    .get_credentials(&2)
                    .await
                    .expect("healthy initial key"),
                Some(pair("healthy", "valid")),
                "{scenario}"
            );
            assert_eq!(reader.get_credentials(&3).await.expect("unknown key"), None);
            tokio::fs::write(&path, "repaired")
                .await
                .expect("repair credential source");
            wait_for(&reader, &1, pair("broken", "repaired")).await;
        }
    }

    #[tokio::test]
    async fn absent_password_is_distinct_from_empty() {
        let dir = tempdir().expect("temporary directory");
        let username = dir.path().join("username");
        let empty_password = dir.path().join("password");
        tokio::fs::write(&username, "root")
            .await
            .expect("username file");
        tokio::fs::write(&empty_password, "")
            .await
            .expect("empty password file");
        for (scenario, password, expected) in [
            ("absent", None, None),
            (
                "empty inline",
                Some(CredentialValue::Inline(String::new())),
                Some(String::new()),
            ),
            (
                "empty file",
                Some(CredentialValue::File(empty_password)),
                Some(String::new()),
            ),
        ] {
            let reader = FileCredentialReader::new(
                HashMap::from([(
                    1,
                    CredentialSources {
                        username: CredentialValue::File(username.clone()),
                        password,
                    },
                )]),
                Duration::from_secs(60),
            )
            .await
            .expect("reader");
            assert_eq!(
                reader.get_credentials(&1).await.expect("lookup"),
                Some(UsernamePassword {
                    username: "root".into(),
                    password: expected
                }),
                "{scenario}"
            );
        }
    }

    #[tokio::test]
    async fn zero_poll_interval_is_rejected() {
        assert!(
            FileCredentialReader::<i32>::new(HashMap::new(), Duration::ZERO)
                .await
                .is_err()
        );
    }

    #[test]
    fn credential_debug_redacts_values() {
        for (scenario, rendered) in [
            (
                "inline source",
                format!("{:?}", CredentialValue::Inline("sensitive".into())),
            ),
            (
                "optional credential",
                format!("{:?}", pair("root", "sensitive")),
            ),
        ] {
            assert!(!rendered.contains("sensitive"), "{scenario}");
        }
    }

    fn pair(username: &str, password: &str) -> UsernamePassword<Option<String>> {
        UsernamePassword {
            username: username.into(),
            password: Some(password.into()),
        }
    }

    async fn wait_for(
        reader: &FileCredentialReader<i32>,
        key: &i32,
        expected: UsernamePassword<Option<String>>,
    ) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if matches!(reader.get_credentials(key).await, Ok(Some(actual)) if actual == expected) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("credentials reload");
    }
}
