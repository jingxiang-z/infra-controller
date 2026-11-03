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

//! Endpoint source backed by an HTTP inventory server.
//!
//! Each discovery pass lists targets with `GET /v1/targets`. Each BMC client
//! fetches credentials with `POST /v1/targets/{mac}/credentials`; a fetch after
//! the first one is the client's refresh after a rejection, so it reports the
//! rejected token and the server can replace that session.

use std::collections::{BTreeMap, HashMap};
use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mac_address::MacAddress;
use nv_redfish::bmc_http::reqwest::Client as ReqwestClient;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::HealthError;
use crate::bmc::{
    BmcClient, BmcLatencyInstrumentation, CredentialProvider, bmc_latency_endpoint_labels,
};
use crate::config::HttpEndpointSourceConfig;
use crate::endpoint::{BmcAddr, BmcCredentials, BmcEndpoint, BoxFuture, EndpointSource};
use crate::metrics::BmcLatencyMetrics;

/// Bounds each inventory server request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Host used in request URLs when the server listens on a Unix socket.
const UNIX_SOCKET_BASE_URL: &str = "http://localhost/";

#[derive(Debug, Deserialize)]
struct TargetRecord {
    mac: String,
    ip: IpAddr,
    #[serde(default)]
    port: Option<u16>,
    #[serde(default)]
    labels: BTreeMap<String, String>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum FetchReason {
    CacheMiss,
    Rejected,
}

#[derive(Serialize)]
struct CredentialRequest<'a> {
    reason: FetchReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<&'a str>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CredentialResponse {
    Session {
        token: String,
    },
    Basic {
        username: String,
        #[serde(default)]
        password: Option<String>,
    },
}

/// The inventory server's base URL and the HTTP client that reaches it.
struct InventoryApi {
    http: HttpClient,
    base_url: Url,
}

impl InventoryApi {
    fn new(cfg: &HttpEndpointSourceConfig) -> Result<Self, HealthError> {
        let mut builder = HttpClient::builder().timeout(REQUEST_TIMEOUT);
        let base_url = match (&cfg.socket_path, &cfg.url) {
            (Some(socket_path), None) => {
                builder = builder.unix_socket(socket_path.as_path());
                Url::parse(UNIX_SOCKET_BASE_URL).expect("static URL parses")
            }
            (None, Some(url)) => url.clone(),
            _ => {
                return Err(HealthError::GenericError(
                    "http endpoint source requires exactly one of `url` or `socket_path`".into(),
                ));
            }
        };
        let http = builder.build().map_err(|error| {
            HealthError::GenericError(format!("http endpoint source client build failed: {error}"))
        })?;
        Ok(Self { http, base_url })
    }

    fn url(&self, path: &str) -> Result<Url, HealthError> {
        self.base_url.join(path).map_err(|error| {
            HealthError::GenericError(format!("invalid inventory server URL for {path}: {error}"))
        })
    }

    async fn targets(&self) -> Result<Vec<TargetRecord>, HealthError> {
        let response = self
            .http
            .get(self.url("v1/targets")?)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| {
                HealthError::GenericError(format!("failed to list inventory targets: {error}"))
            })?;
        response.json().await.map_err(|error| {
            HealthError::GenericError(format!("invalid inventory target list: {error}"))
        })
    }

    async fn credentials(
        &self,
        mac: MacAddress,
        request: &CredentialRequest<'_>,
    ) -> Result<BmcCredentials, HealthError> {
        // Errors omit the response body, which may contain credentials.
        let response = self
            .http
            .post(self.url(&format!("v1/targets/{mac}/credentials"))?)
            .json(request)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| {
                HealthError::GenericError(format!(
                    "failed to fetch BMC credentials from inventory server: {}",
                    error.without_url()
                ))
            })?;
        let credentials = response.json::<CredentialResponse>().await.map_err(|_| {
            HealthError::GenericError("invalid credential response from inventory server".into())
        })?;
        Ok(match credentials {
            CredentialResponse::Session { token } => BmcCredentials::SessionToken { token },
            CredentialResponse::Basic { username, password } => {
                BmcCredentials::UsernamePassword { username, password }
            }
        })
    }
}

/// Credentials for one BMC, fetched from the inventory server.
struct HttpCredentialProvider {
    api: Arc<InventoryApi>,
    mac: MacAddress,
    /// The last credentials issued to this BMC's client. The client fetches
    /// again only after the BMC rejects them, so a later fetch reports them.
    issued: tokio::sync::Mutex<Option<BmcCredentials>>,
}

impl CredentialProvider for HttpCredentialProvider {
    fn fetch_credentials<'a>(
        &'a self,
        _endpoint: &'a BmcAddr,
    ) -> BoxFuture<'a, Result<BmcCredentials, HealthError>> {
        Box::pin(async move {
            let mut issued = self.issued.lock().await;
            let request = match issued.as_ref() {
                None => CredentialRequest {
                    reason: FetchReason::CacheMiss,
                    token: None,
                },
                Some(BmcCredentials::SessionToken { token }) => CredentialRequest {
                    reason: FetchReason::Rejected,
                    token: Some(token),
                },
                Some(BmcCredentials::UsernamePassword { .. }) => CredentialRequest {
                    reason: FetchReason::Rejected,
                    token: None,
                },
            };
            let credentials = self.api.credentials(self.mac, &request).await?;
            *issued = Some(credentials.clone());
            Ok(credentials)
        })
    }
}

/// A BMC client kept across discovery passes, so running collectors keep it.
struct CachedClient {
    bmc: Arc<BmcClient>,
    ip: IpAddr,
    port: Option<u16>,
}

pub struct HttpEndpointSource {
    api: Arc<InventoryApi>,
    reqwest: ReqwestClient,
    proxy_url: Option<Url>,
    cache_size: usize,
    bmc_request_concurrency: NonZeroUsize,
    bmc_latency_metrics: Option<Arc<BmcLatencyMetrics>>,
    /// Clients by BMC MAC. A new address for the same MAC gets a new client.
    clients: Mutex<HashMap<MacAddress, CachedClient>>,
}

impl HttpEndpointSource {
    pub(crate) fn from_config_with_request_concurrency(
        cfg: &HttpEndpointSourceConfig,
        reqwest: &ReqwestClient,
        proxy_url: Option<&Url>,
        cache_size: usize,
        bmc_request_concurrency: NonZeroUsize,
        bmc_latency_metrics: Option<Arc<BmcLatencyMetrics>>,
    ) -> Result<Self, HealthError> {
        Ok(Self {
            api: Arc::new(InventoryApi::new(cfg)?),
            reqwest: reqwest.clone(),
            proxy_url: proxy_url.cloned(),
            cache_size,
            bmc_request_concurrency,
            bmc_latency_metrics,
            clients: Mutex::new(HashMap::new()),
        })
    }

    async fn load_endpoints(&self) -> Result<Vec<Arc<BmcEndpoint>>, HealthError> {
        let targets = self.api.targets().await?;
        let endpoints = self.build_endpoints(targets);
        tracing::debug!(
            endpoint_count = endpoints.len(),
            "Loaded http inventory endpoints"
        );
        Ok(endpoints)
    }

    fn build_endpoints(&self, targets: Vec<TargetRecord>) -> Vec<Arc<BmcEndpoint>> {
        let mut clients = self.clients.lock().expect("http clients mutex poisoned");
        let mut endpoints = Vec::with_capacity(targets.len());
        let mut seen = HashMap::with_capacity(targets.len());

        for target in targets {
            let mac = match target.mac.parse::<MacAddress>() {
                Ok(mac) => mac,
                Err(error) => {
                    tracing::warn!(?error, bmc_mac_address = %target.mac, "Invalid MAC in inventory target; skipping");
                    continue;
                }
            };
            if seen.insert(mac, ()).is_some() {
                tracing::warn!(bmc_mac_address = %mac, "Duplicate MAC in inventory targets; skipping");
                continue;
            }

            let addr = BmcAddr {
                ip: target.ip,
                port: target.port,
                mac: Some(mac),
            };
            let bmc = match clients.get(&mac) {
                Some(cached) if cached.ip == target.ip && cached.port == target.port => {
                    cached.bmc.clone()
                }
                _ => {
                    let provider = Arc::new(HttpCredentialProvider {
                        api: self.api.clone(),
                        mac,
                        issued: tokio::sync::Mutex::new(None),
                    });
                    let instrumentation = self.bmc_latency_metrics.clone().map(|metrics| {
                        BmcLatencyInstrumentation::new(
                            metrics,
                            bmc_latency_endpoint_labels(None, None),
                        )
                    });
                    let bmc = match BmcClient::new(
                        self.reqwest.clone(),
                        addr.clone(),
                        provider,
                        self.proxy_url.clone(),
                        self.cache_size,
                        self.bmc_request_concurrency,
                        instrumentation,
                    ) {
                        Ok(client) => Arc::new(client),
                        Err(error) => {
                            tracing::warn!(?error, bmc_address = ?addr, "Failed to construct BmcClient for inventory target; skipping");
                            continue;
                        }
                    };
                    clients.insert(
                        mac,
                        CachedClient {
                            bmc: bmc.clone(),
                            ip: target.ip,
                            port: target.port,
                        },
                    );
                    bmc
                }
            };

            endpoints.push(Arc::new(BmcEndpoint {
                addr,
                metadata: None,
                rack_id: None,
                labels: target.labels,
                bmc,
            }));
        }

        clients.retain(|mac, _| seen.contains_key(mac));
        endpoints
    }
}

impl EndpointSource for HttpEndpointSource {
    fn fetch_bmc_hosts<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Arc<BmcEndpoint>>, HealthError>> {
        Box::pin(self.load_endpoints())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex as StdMutex;

    use axum::Json;
    use axum::extract::{Path, State};
    use axum::routing::{get, post};
    use serde_json::{Value, json};

    use super::*;
    use crate::endpoint::test_support::reqwest;

    #[derive(Clone, Default)]
    struct Server {
        targets: Arc<StdMutex<Value>>,
        requests: Arc<StdMutex<Vec<(String, Value)>>>,
    }

    fn serve(server: Server) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create socket dir");
        let socket_path = dir.path().join("api.sock");
        let listener = tokio::net::UnixListener::bind(&socket_path).expect("bind socket");
        let app = axum::Router::new()
            .route(
                "/v1/targets",
                get(|State(server): State<Server>| async move {
                    Json(server.targets.lock().unwrap().clone())
                }),
            )
            .route(
                "/v1/targets/{mac}/credentials",
                post(
                    |State(server): State<Server>,
                     Path(mac): Path<String>,
                     Json(body): Json<Value>| async move {
                        let mut requests = server.requests.lock().unwrap();
                        requests.push((mac, body));
                        Json(json!({"type": "session", "token": format!("token-{}", requests.len())}))
                    },
                ),
            )
            .with_state(server);
        tokio::spawn(async move { axum::serve(listener, app).await });
        (dir, socket_path)
    }

    fn source(socket_path: PathBuf) -> HttpEndpointSource {
        HttpEndpointSource::from_config_with_request_concurrency(
            &HttpEndpointSourceConfig {
                url: None,
                socket_path: Some(socket_path),
            },
            &reqwest(),
            None,
            10,
            NonZeroUsize::MIN,
            None,
        )
        .expect("source builds")
    }

    #[tokio::test]
    async fn lists_targets_and_keeps_clients_until_the_address_changes() {
        let server = Server::default();
        *server.targets.lock().unwrap() = json!([
            {"mac": "aa:bb:cc:00:00:01", "ip": "10.0.0.1", "labels": {"rack": "r1"}},
            {"mac": "not-a-mac", "ip": "10.0.0.2"},
            {"mac": "aa:bb:cc:00:00:01", "ip": "10.0.0.3"},
        ]);
        let (_dir, socket_path) = serve(server.clone());
        let source = source(socket_path);

        let first = source.fetch_bmc_hosts().await.expect("first pass");
        assert_eq!(first.len(), 1, "invalid and duplicate MACs are skipped");
        assert_eq!(first[0].addr.ip, "10.0.0.1".parse::<IpAddr>().unwrap());
        assert_eq!(first[0].labels.get("rack").map(String::as_str), Some("r1"));

        let second = source.fetch_bmc_hosts().await.expect("second pass");
        assert!(Arc::ptr_eq(&first[0].bmc, &second[0].bmc));

        *server.targets.lock().unwrap() =
            json!([{"mac": "aa:bb:cc:00:00:01", "ip": "10.0.0.9", "port": 8443}]);
        let moved = source.fetch_bmc_hosts().await.expect("third pass");
        assert_eq!(moved[0].addr.port, Some(8443));
        assert!(!Arc::ptr_eq(&first[0].bmc, &moved[0].bmc));
    }

    #[tokio::test]
    async fn later_fetches_report_the_rejected_token() {
        let server = Server::default();
        let (_dir, socket_path) = serve(server.clone());
        let source = source(socket_path);
        let mac: MacAddress = "aa:bb:cc:00:00:01".parse().unwrap();
        let provider = HttpCredentialProvider {
            api: source.api.clone(),
            mac,
            issued: tokio::sync::Mutex::new(None),
        };
        let addr = BmcAddr {
            ip: "10.0.0.1".parse().unwrap(),
            port: None,
            mac: Some(mac),
        };

        let first = provider
            .fetch_credentials(&addr)
            .await
            .expect("first fetch");
        let second = provider.fetch_credentials(&addr).await.expect("refresh");

        assert!(
            first
                == BmcCredentials::SessionToken {
                    token: "token-1".into()
                }
        );
        assert!(
            second
                == BmcCredentials::SessionToken {
                    token: "token-2".into()
                }
        );
        let requests = server.requests.lock().unwrap().clone();
        assert_eq!(
            requests,
            vec![
                (
                    "AA:BB:CC:00:00:01".to_string(),
                    json!({"reason": "cache_miss"})
                ),
                (
                    "AA:BB:CC:00:00:01".to_string(),
                    json!({"reason": "rejected", "token": "token-1"})
                ),
            ]
        );
    }
}
