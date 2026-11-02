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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use carbide_uuid::rack::RackId;
use mac_address::MacAddress;
use nv_redfish::bmc_http::reqwest::Client as ReqwestClient;
use reqwest::Client as HttpClient;
use serde::Deserialize;
use serde_json::json;
use url::Url;

use crate::HealthError;
use crate::bmc::{
    BmcClient, BmcLatencyEndpointLabels, BmcLatencyInstrumentation, CredentialProvider,
    bmc_latency_endpoint_labels,
};
use crate::config::{
    ClusterEndpointSourceConfig, StaticBmcEndpoint, StaticMachineEndpoint,
    StaticPowerShelfEndpoint, StaticSwitchEndpoint, StaticSwitchEndpointRole,
};
use crate::endpoint::{
    BmcAddr, BmcCredentials, BmcEndpoint, BoxFuture, EndpointMetadata, EndpointSource,
    SharedSystemUuid, static_endpoint_metadata,
};
use crate::metrics::BmcLatencyMetrics;
use crate::static_credentials::CredentialFile;

// ── Inventory file shape ──────────────────────────────────────────────────────
//
// The same file can be shared with the OOB collector. Entries are grouped by
// collection. With `credentials_file`, each collection selects the credential
// profile of the same name and every entry needs its real MAC. Without it, the
// embedded `default_credentials` apply to every entry, and an entry without a
// MAC keeps the MAC synthesized from its IPv4 address. The original `hostname`
// and `bmc_ip` field names remain accepted.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileInventory {
    /// Credentials for every entry when no credentials file is configured;
    /// ignored otherwise.
    #[serde(default)]
    default_credentials: Option<FileCredentials>,
    #[serde(default)]
    nodes: Vec<FileEntry>,
    #[serde(default)]
    power_shelves: Vec<FileEntry>,
    #[serde(default)]
    switches: Vec<FileEntry>,
    #[serde(default)]
    switch_hosts: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct FileCredentials {
    username: String,
    password: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileEntry {
    #[serde(default)]
    mac: Option<MacAddress>,
    #[serde(alias = "bmc_ip")]
    address: IpAddr,
    #[serde(default)]
    port: Option<u16>,
    #[serde(default, alias = "hostname")]
    name: Option<String>,
    #[serde(default)]
    rack: Option<String>,
    #[serde(default)]
    labels: BTreeMap<String, String>,
    #[serde(default)]
    machine: Option<StaticMachineEndpoint>,
    #[serde(default)]
    power_shelf: Option<StaticPowerShelfEndpoint>,
    #[serde(default)]
    switch: Option<FileSwitch>,
}

/// Switch metadata; the collection sets the endpoint role.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSwitch {
    id: Option<String>,
    serial: Option<String>,
    #[serde(alias = "physical_slot_number")]
    slot_number: Option<i32>,
    #[serde(alias = "compute_tray_index")]
    tray_index: Option<i32>,
    nvlink_domain_uuid: Option<String>,
    #[serde(default)]
    is_primary: bool,
    #[serde(default)]
    nmxc_enabled: Option<bool>,
    #[serde(default)]
    nmxt_enabled: Option<bool>,
}

impl FileSwitch {
    fn into_static(self, endpoint_role: StaticSwitchEndpointRole) -> StaticSwitchEndpoint {
        StaticSwitchEndpoint {
            id: self.id,
            serial: self.serial,
            slot_number: self.slot_number,
            tray_index: self.tray_index,
            nvlink_domain_uuid: self.nvlink_domain_uuid,
            endpoint_role,
            is_primary: self.is_primary,
            nmxc_enabled: self.nmxc_enabled,
            nmxt_enabled: self.nmxt_enabled,
        }
    }
}

/// An inventory collection: its name is also its credential profile.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Collection {
    Nodes,
    PowerShelves,
    Switches,
    SwitchHosts,
}

impl Collection {
    fn name(self) -> &'static str {
        match self {
            Self::Nodes => "nodes",
            Self::PowerShelves => "power_shelves",
            Self::Switches => "switches",
            Self::SwitchHosts => "switch_hosts",
        }
    }

    /// The metadata field entries in this collection may carry.
    fn metadata_field(self) -> &'static str {
        match self {
            Self::Nodes => "machine",
            Self::PowerShelves => "power_shelf",
            Self::Switches | Self::SwitchHosts => "switch",
        }
    }
}

// ── Canonical internal node shape (both paths produce this) ──────────────────

struct ClusterNode {
    hostname: String,
    bmc_ip: IpAddr,
    port: Option<u16>,
    /// The BMC MAC; `None` synthesizes one from an IPv4 address.
    mac: Option<MacAddress>,
    rack: Option<String>,
    labels: BTreeMap<String, String>,
    metadata: Option<EndpointMetadata>,
    credentials: BmcCredentials,
}

// ── Cluster Manager JSON RPC ────────────────────────────────────────────────
//
// The cluster manager exposes a JSON RPC API at /json/ for inventory and credential queries.
// Two calls are made:
//
//   1. cmdevice.getDevices — inventory: one entry per PhysicalNode with hostname
//      and BMC interface IP.
//
//   2. cmpart.getPartition("<partition>") — credentials: the bmcsettings sub-object
//      holds the cluster-wide BMC username and password (stored by the cluster manager daemon).
//      Default partition is "base".
//
// Exact call names and response field paths need verification against the live
// head node:
//   GET  https://<head-node>:8081/api           — lists available services + calls
//   POST https://<head-node>:8081/json/          — JSON RPC endpoint
//
// When Joab confirms the real call names, update MANAGER_DEVICE_CALL and
// MANAGER_PARTITION_CALL below, and the field extractors.

const MANAGER_DEVICE_CALL: &str = "getDevices";
const MANAGER_PARTITION_CALL: &str = "getPartition";

fn build_http() -> Result<HttpClient, HealthError> {
    HttpClient::builder().build().map_err(|e| {
        HealthError::GenericError(format!("Cluster manager HTTP client build failed: {e}"))
    })
}

async fn manager_rpc(
    http: &HttpClient,
    json_rpc_url: &Url,
    service: &str,
    call: &str,
    arg: serde_json::Value,
) -> Result<serde_json::Value, HealthError> {
    let body = json!({ "service": service, "call": call, "arg": arg });

    tracing::debug!(service, call, url = %json_rpc_url, "cluster manager JSON RPC call");

    let response = http
        .post(json_rpc_url.as_str())
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            HealthError::GenericError(format!("cluster manager RPC {service}.{call} failed: {e}"))
        })?;

    if !response.status().is_success() {
        return Err(HealthError::GenericError(format!(
            "cluster manager RPC {service}.{call} returned HTTP {}",
            response.status()
        )));
    }

    response.json().await.map_err(|e| {
        HealthError::GenericError(format!(
            "cluster manager RPC {service}.{call} parse failed: {e}"
        ))
    })
}

/// Fetches the partition's BMC credentials. A response without them falls back
/// to the configured defaults; a failed call is returned as an error so the
/// caller can keep the credentials it fetched last.
async fn fetch_manager_credentials(
    http: &HttpClient,
    json_rpc_url: &Url,
    cfg: &ClusterEndpointSourceConfig,
) -> Result<(String, Option<String>), HealthError> {
    // cmsh equivalent: partition use <partition> → bmcsettings → get username / get password
    // Password stored by the cluster manager daemon at partition level.
    let result = manager_rpc(
        http,
        json_rpc_url,
        "cmpart",
        MANAGER_PARTITION_CALL,
        json!(cfg.cluster_manager_partition),
    )
    .await;

    let raw = result?;

    tracing::debug!("Raw cluster manager partition response received");

    // Navigate to bmcsettings — try likely paths.
    // Update once live /api confirms the exact field layout.
    let bmc_settings = raw
        .pointer("/bmcSettings")
        .or_else(|| raw.pointer("/result/bmcSettings"))
        .or_else(|| raw.pointer("/data/bmcSettings"));

    let username = bmc_settings
        .and_then(|s| s.get("username").or_else(|| s.get("userName")))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            tracing::debug!(
                "Cluster manager partition response missing bmcSettings.username; using config fallback. \
                 Probe /api on head node to find correct field path."
            );
            cfg.default_username.clone()
        });

    let password = bmc_settings
        .and_then(|s| s.get("password").or_else(|| s.get("bmcPassword")))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| {
            tracing::debug!(
                "Cluster manager partition response missing bmcSettings.password; using config fallback."
            );
            cfg.default_password.clone()
        });

    Ok((username, password))
}

fn extract_manager_devices(
    raw: &serde_json::Value,
    username: String,
    password: Option<String>,
) -> Result<Vec<ClusterNode>, HealthError> {
    // The device list may be a top-level array or wrapped in result/data/items.
    // Update once live /api confirms the exact response shape.
    let items: Vec<&serde_json::Value> = if let Some(arr) = raw.as_array() {
        arr.iter().collect()
    } else {
        match ["result", "data", "items", "devices"]
            .iter()
            .find_map(|key| raw.get(key).and_then(|v| v.as_array()))
        {
            Some(arr) => arr.iter().collect(),
            None => {
                return Err(HealthError::GenericError(
                    "Cluster manager getDevices response has no recognized shape; \
                     probe /api on head node and update extract_manager_devices in cluster.rs"
                        .to_string(),
                ));
            }
        }
    };

    let item_count = items.len();
    let mut nodes = Vec::new();
    for item in items {
        let hostname = item
            .get("hostname")
            .or_else(|| item.get("name"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // BMC IP lives in the BMC interface, not at the top level of the device.
        // cmsh: device → interfaces → use ipmi0/ilo0/rf0 → get ip
        // Try common field paths; update once confirmed.
        let bmc_ip_str = item
            .pointer("/interfaces/ipmi0/ip")
            .or_else(|| item.pointer("/interfaces/bmc/ip"))
            .or_else(|| item.pointer("/interfaces/ilo0/ip"))
            .or_else(|| item.pointer("/interfaces/rf0/ip"))
            .or_else(|| item.get("bmcAddress"))
            .or_else(|| item.get("bmcIp"))
            .or_else(|| item.get("ipmiAddress"))
            .and_then(|v| v.as_str());

        // Cluster manager category maps to our rack identifier.
        let rack = item
            .get("category")
            .or_else(|| item.get("partition"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let (Some(hostname), Some(bmc_ip_str)) = (hostname, bmc_ip_str) else {
            tracing::debug!(
                item_keys = ?item.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()),
                rack_id = rack.as_deref().map(tracing::field::display),
                "Cluster manager device entry missing hostname or BMC IP; skipping. \
                 Update field paths in extract_manager_devices once head node is probed."
            );
            continue;
        };

        let bmc_ip: IpAddr = match bmc_ip_str.parse() {
            Ok(ip) => ip,
            Err(e) => {
                tracing::warn!(
                    hostname,
                    bmc_ip_address = bmc_ip_str,
                    rack_id = rack.as_deref().map(tracing::field::display),
                    error = %e,
                    "Invalid BMC IP; skipping"
                );
                continue;
            }
        };

        nodes.push(ClusterNode {
            hostname,
            bmc_ip,
            port: None,
            mac: None,
            rack,
            labels: BTreeMap::new(),
            metadata: None,
            credentials: BmcCredentials::UsernamePassword {
                username: username.clone(),
                password: password.clone(),
            },
        });
    }

    if nodes.is_empty() && item_count > 0 {
        return Err(HealthError::GenericError(
            "Cluster manager getDevices returned entries but none had a recognized hostname or BMC IP; \
             probe /api on head node and update extract_manager_devices in cluster.rs"
                .to_string(),
        ));
    }

    Ok(nodes)
}

// ── Source implementations ────────────────────────────────────────────────────

/// Supplies the credentials the cluster inventory last listed for one BMC.
///
/// Discovery updates them on every pass. The BMC client asks again after an
/// authentication failure, so running collectors pick up rotated credentials
/// without a restart.
struct InventoryCredentialProvider {
    credentials: ArcSwap<BmcCredentials>,
}

impl InventoryCredentialProvider {
    fn new(credentials: BmcCredentials) -> Self {
        Self {
            credentials: ArcSwap::from_pointee(credentials),
        }
    }

    /// Installs `credentials` and reports whether they differ from the current ones.
    fn update(&self, credentials: BmcCredentials) -> bool {
        if **self.credentials.load() == credentials {
            return false;
        }
        self.credentials.store(Arc::new(credentials));
        true
    }
}

impl CredentialProvider for InventoryCredentialProvider {
    fn fetch_credentials<'a>(
        &'a self,
        _endpoint: &'a BmcAddr,
    ) -> BoxFuture<'a, Result<BmcCredentials, HealthError>> {
        let credentials = (**self.credentials.load()).clone();
        Box::pin(async move { Ok(credentials) })
    }
}

/// A BMC client kept across discovery passes, so collectors that captured it
/// see credential updates.
struct CachedClient {
    bmc: Arc<BmcClient>,
    provider: Arc<InventoryCredentialProvider>,
    /// The address the client was built for; a new port or MAC needs a new client.
    addr: BmcAddr,
    /// The machine and rack in the client's latency labels; a change needs a new client.
    latency_labels: BmcLatencyEndpointLabels,
    /// Carried over when the client is rebuilt for the same MAC.
    system_uuid: SharedSystemUuid,
}

pub struct ClusterEndpointSource {
    cfg: ClusterEndpointSourceConfig,
    reqwest: ReqwestClient,
    proxy_url: Option<Url>,
    cache_size: usize,
    bmc_request_concurrency: NonZeroUsize,
    bmc_latency_metrics: Option<Arc<BmcLatencyMetrics>>,
    /// Resolves each file entry's credentials by MAC and collection profile.
    credentials_file: Option<Arc<CredentialFile>>,
    /// Clients by BMC IP. A row's IP identifies its BMC within one inventory.
    clients: Mutex<HashMap<IpAddr, CachedClient>>,
    /// Inventory rows skipped for a repeated BMC IP in the last pass, so each
    /// is reported once rather than on every pass.
    duplicate_rows: Mutex<HashSet<(IpAddr, String)>>,
    /// The partition credentials from the last successful cluster manager call.
    manager_credentials: Mutex<Option<(String, Option<String>)>>,
}

impl ClusterEndpointSource {
    pub fn from_config(
        cfg: ClusterEndpointSourceConfig,
        reqwest: &ReqwestClient,
        proxy_url: Option<&Url>,
        cache_size: usize,
        bmc_latency_metrics: Option<Arc<BmcLatencyMetrics>>,
    ) -> Self {
        Self::from_config_with_request_concurrency(
            cfg,
            reqwest,
            proxy_url,
            cache_size,
            NonZeroUsize::MIN,
            bmc_latency_metrics,
            None,
        )
    }

    pub(crate) fn from_config_with_request_concurrency(
        cfg: ClusterEndpointSourceConfig,
        reqwest: &ReqwestClient,
        proxy_url: Option<&Url>,
        cache_size: usize,
        bmc_request_concurrency: NonZeroUsize,
        bmc_latency_metrics: Option<Arc<BmcLatencyMetrics>>,
        credentials_file: Option<Arc<CredentialFile>>,
    ) -> Self {
        Self {
            cfg,
            credentials_file,
            reqwest: reqwest.clone(),
            proxy_url: proxy_url.cloned(),
            cache_size,
            bmc_request_concurrency,
            bmc_latency_metrics,
            clients: Mutex::new(HashMap::new()),
            duplicate_rows: Mutex::new(HashSet::new()),
            manager_credentials: Mutex::new(None),
        }
    }

    async fn load_endpoints(&self) -> Result<Vec<Arc<BmcEndpoint>>, HealthError> {
        let nodes = if let Some(ref cluster_manager_url) = self.cfg.cluster_manager_url {
            self.fetch_from_manager(cluster_manager_url).await?
        } else {
            read_from_file(&self.cfg, self.credentials_file.as_deref())?
        };
        let endpoints = self.build_endpoints(nodes);
        tracing::info!(endpoint_count = endpoints.len(), "Loaded cluster endpoints");
        Ok(endpoints)
    }

    async fn fetch_from_manager(
        &self,
        cluster_manager_url: &Url,
    ) -> Result<Vec<ClusterNode>, HealthError> {
        let cfg = &self.cfg;
        let http = build_http()?;
        let json_rpc_url = cluster_manager_url
            .join("/json/")
            .map_err(|e| HealthError::GenericError(format!("Invalid cluster manager URL: {e}")))?;

        tracing::info!(url = %json_rpc_url, partition = %cfg.cluster_manager_partition, "Fetching cluster inventory from cluster manager");

        // Call 1: partition-level BMC credentials
        let (username, password) = self.manager_credentials(&http, &json_rpc_url).await;

        // Call 2: device inventory
        let raw = manager_rpc(
            &http,
            &json_rpc_url,
            "cmdevice",
            MANAGER_DEVICE_CALL,
            json!({"type": "PhysicalNode"}),
        )
        .await?;

        tracing::debug!(
            response_keys = ?raw.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()),
            response_is_array = raw.is_array(),
            "Raw cluster manager device response received"
        );

        let nodes = extract_manager_devices(&raw, username, password)?;
        tracing::info!(
            loaded_node_count = nodes.len(),
            "Cluster manager device fetch complete"
        );
        Ok(nodes)
    }

    /// Returns the partition credentials. When the call fails, the credentials
    /// from the last successful call are kept, so one failed call does not
    /// swap every client onto the configured defaults and back.
    async fn manager_credentials(
        &self,
        http: &HttpClient,
        json_rpc_url: &Url,
    ) -> (String, Option<String>) {
        match fetch_manager_credentials(http, json_rpc_url, &self.cfg).await {
            Ok(credentials) => {
                *self
                    .manager_credentials
                    .lock()
                    .expect("manager credentials mutex poisoned") = Some(credentials.clone());
                credentials
            }
            Err(e) => {
                let last = self
                    .manager_credentials
                    .lock()
                    .expect("manager credentials mutex poisoned")
                    .clone();
                match last {
                    Some(credentials) => {
                        tracing::warn!(
                            error = %e,
                            partition = %self.cfg.cluster_manager_partition,
                            "Failed to fetch cluster manager partition credentials; keeping the last fetched credentials"
                        );
                        credentials
                    }
                    None => {
                        tracing::warn!(
                            error = %e,
                            partition = %self.cfg.cluster_manager_partition,
                            "Failed to fetch cluster manager partition credentials; using config fallback"
                        );
                        (
                            self.cfg.default_username.clone(),
                            self.cfg.default_password.clone(),
                        )
                    }
                }
            }
        }
    }

    /// Builds this pass's endpoints, reusing each BMC's client from earlier
    /// passes and updating its credentials in place. An address or latency-label
    /// change (machine or rack) replaces the client.
    fn build_endpoints(&self, nodes: Vec<ClusterNode>) -> Vec<Arc<BmcEndpoint>> {
        let mut endpoints = Vec::with_capacity(nodes.len());
        let mut clients = self.clients.lock().expect("cluster client mutex poisoned");
        // The endpoint key derives one-to-one from the BMC IP, so a repeated IP
        // would map two inventory rows onto one set of collectors. Maps each IP
        // to the row whose endpoint was built.
        let mut built: HashMap<IpAddr, usize> = HashMap::with_capacity(nodes.len());
        let mut duplicate_rows = HashSet::new();
        let reported_duplicates = std::mem::take(
            &mut *self
                .duplicate_rows
                .lock()
                .expect("duplicate rows mutex poisoned"),
        );

        for (index, node) in nodes.iter().enumerate() {
            if let Some(&first) = built.get(&node.bmc_ip) {
                let row = (node.bmc_ip, node.hostname.clone());
                if !reported_duplicates.contains(&row) {
                    let first = &nodes[first];
                    tracing::warn!(
                        bmc_ip_address = %node.bmc_ip,
                        hostname = %node.hostname,
                        rack_id = node.rack.as_deref(),
                        first_hostname = %first.hostname,
                        first_rack_id = first.rack.as_deref(),
                        "Skipping cluster inventory row with duplicate BMC IP; keeping the first row"
                    );
                }
                duplicate_rows.insert(row);
                continue;
            }

            let rack_id = node.rack.as_deref().map(RackId::new);
            let mac = node.mac.or(match node.bmc_ip {
                IpAddr::V4(v4) => {
                    // Retain the synthetic MAC already used in IPv4 logs and metric labels.
                    let [o1, o2, o3, o4] = v4.octets();
                    Some(MacAddress::new([0x02, 0x00, o1, o2, o3, o4]))
                }
                // IPv6 needs the full address; a synthetic MAC cannot distinguish all IPv6 BMCs.
                IpAddr::V6(_) => None,
            });

            let addr = BmcAddr {
                ip: node.bmc_ip,
                port: node.port.or(self.cfg.port),
                mac,
            };
            let latency_labels =
                bmc_latency_endpoint_labels(node.metadata.as_ref(), rack_id.as_ref());
            let mut metadata = node.metadata.clone();

            let cached = clients.get(&node.bmc_ip);
            let bmc = match cached {
                Some(cached) if cached.addr == addr && cached.latency_labels == latency_labels => {
                    if cached.provider.update(node.credentials.clone()) {
                        cached.bmc.credentials_changed();
                        tracing::info!(
                            hostname = %node.hostname,
                            bmc_ip_address = %node.bmc_ip,
                            "BMC credentials changed in cluster inventory; running collectors use them after their next authentication failure"
                        );
                    }
                    cached.bmc.clone()
                }
                _ => {
                    let bmc_latency_instrumentation =
                        self.bmc_latency_metrics.clone().map(|metrics| {
                            BmcLatencyInstrumentation::new(metrics, latency_labels.clone())
                        });
                    let provider =
                        Arc::new(InventoryCredentialProvider::new(node.credentials.clone()));
                    let bmc = match BmcClient::new(
                        self.reqwest.clone(),
                        addr.clone(),
                        provider.clone(),
                        self.proxy_url.clone(),
                        self.cache_size,
                        self.bmc_request_concurrency,
                        bmc_latency_instrumentation,
                    ) {
                        Ok(c) => Arc::new(c),
                        Err(e) => {
                            tracing::warn!(
                                error = ?e,
                                hostname = %node.hostname,
                                rack_id = rack_id.as_ref().map(tracing::field::display),
                                "Failed to construct BmcClient for cluster endpoint; skipping"
                            );
                            continue;
                        }
                    };
                    // The same MAC is the same hardware, so its system UUID carries over.
                    let system_uuid = cached
                        .filter(|cached| cached.addr.mac.is_some() && cached.addr.mac == mac)
                        .map(|cached| cached.system_uuid.clone())
                        .unwrap_or_default();
                    clients.insert(
                        node.bmc_ip,
                        CachedClient {
                            bmc: bmc.clone(),
                            provider,
                            addr: addr.clone(),
                            latency_labels,
                            system_uuid,
                        },
                    );
                    bmc
                }
            };
            if let Some(EndpointMetadata::Machine(machine)) = &mut metadata {
                machine.system_uuid = clients[&node.bmc_ip].system_uuid.clone();
            }

            endpoints.push(Arc::new(BmcEndpoint {
                addr,
                metadata,
                rack_id,
                labels: node.labels.clone(),
                bmc,
            }));
            built.insert(node.bmc_ip, index);
        }

        clients.retain(|ip, _| built.contains_key(ip));
        *self
            .duplicate_rows
            .lock()
            .expect("duplicate rows mutex poisoned") = duplicate_rows;
        endpoints
    }
}

fn read_from_file(
    cfg: &ClusterEndpointSourceConfig,
    credentials_file: Option<&CredentialFile>,
) -> Result<Vec<ClusterNode>, HealthError> {
    let path = cfg.inventory_path.display();
    let contents = std::fs::read_to_string(&cfg.inventory_path).map_err(|e| {
        HealthError::GenericError(format!("Failed to read cluster inventory {path}: {e}"))
    })?;
    let inventory: FileInventory = serde_json::from_str(&contents)
        .map_err(|e| HealthError::GenericError(format!("Invalid cluster inventory {path}: {e}")))?;
    let invalid = |problem: String| {
        HealthError::GenericError(format!("Invalid cluster inventory {path}: {problem}"))
    };

    let default_credentials = match (credentials_file, inventory.default_credentials) {
        (Some(_), Some(_)) => {
            tracing::debug!(
                path = %path,
                "Ignoring cluster inventory default_credentials; a credentials file is configured"
            );
            None
        }
        (Some(_), None) => None,
        (None, Some(credentials)) => Some(BmcCredentials::UsernamePassword {
            username: credentials.username,
            password: credentials.password,
        }),
        (None, None) => {
            return Err(invalid(
                "default_credentials is required without endpoint_sources.cluster.credentials_file"
                    .to_string(),
            ));
        }
    };

    let collections = [
        (Collection::Nodes, inventory.nodes),
        (Collection::PowerShelves, inventory.power_shelves),
        (Collection::Switches, inventory.switches),
        (Collection::SwitchHosts, inventory.switch_hosts),
    ];
    let mut nodes = Vec::new();
    let mut macs: HashMap<MacAddress, (&str, usize)> = HashMap::new();
    for (collection, entries) in collections {
        let name = collection.name();
        for (index, entry) in entries.into_iter().enumerate() {
            let at = format!("{name}[{index}]");
            if let Some(mac) = entry.mac
                && let Some((first, first_index)) = macs.insert(mac, (name, index))
            {
                return Err(invalid(format!(
                    "{at}.mac duplicates {first}[{first_index}].mac"
                )));
            }
            for (field, present) in [
                ("machine", entry.machine.is_some()),
                ("power_shelf", entry.power_shelf.is_some()),
                ("switch", entry.switch.is_some()),
            ] {
                if present && field != collection.metadata_field() {
                    return Err(invalid(format!("{at} does not support {field} metadata")));
                }
            }
            if collection == Collection::SwitchHosts && entry.switch.is_none() {
                return Err(invalid(format!("{at} requires switch metadata")));
            }

            let credentials = match (credentials_file, &default_credentials) {
                (Some(file), _) => {
                    let mac = entry.mac.ok_or_else(|| {
                        invalid(format!("{at} requires mac with a credentials file"))
                    })?;
                    file.resolve(mac, Some(name)).map_err(|e| {
                        invalid(format!(
                            "{at} credentials from {}: {e}",
                            file.path().display()
                        ))
                    })?
                }
                (None, Some(credentials)) => credentials.clone(),
                (None, None) => unreachable!("default credentials are checked above"),
            };

            // Validate and build metadata exactly as for a static endpoint.
            let endpoint_role = if collection == Collection::SwitchHosts {
                StaticSwitchEndpointRole::Host
            } else {
                StaticSwitchEndpointRole::Bmc
            };
            let rack = entry.rack.filter(|rack| !rack.is_empty());
            let endpoint = StaticBmcEndpoint {
                ip: entry.address,
                port: entry.port,
                mac: entry
                    .mac
                    .map_or_else(|| entry.address.to_string(), |mac| mac.to_string()),
                credentials_profile: None,
                machine: entry.machine,
                power_shelf: entry.power_shelf,
                switch: entry.switch.map(|switch| switch.into_static(endpoint_role)),
                rack_id: rack.clone(),
                labels: entry.labels,
            };
            endpoint.validate(name, index).map_err(invalid)?;

            nodes.push(ClusterNode {
                hostname: entry.name.unwrap_or_else(|| entry.address.to_string()),
                bmc_ip: entry.address,
                port: entry.port,
                mac: entry.mac,
                rack,
                metadata: static_endpoint_metadata(&endpoint),
                labels: endpoint.labels,
                credentials,
            });
        }
    }
    Ok(nodes)
}

impl EndpointSource for ClusterEndpointSource {
    fn fetch_bmc_hosts<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Arc<BmcEndpoint>>, HealthError>> {
        Box::pin(self.load_endpoints())
    }
}

#[cfg(test)]
mod tests {
    use carbide_test_support::{Check, check_values};

    use super::*;
    use crate::bmc::CollectorSweep;
    use crate::endpoint::test_support::reqwest;
    use crate::sink::EventContext;

    #[tokio::test]
    async fn file_inventory_preserves_ipv4_keys_and_distinguishes_ipv6_addresses() {
        let cases = [
            Check {
                scenario: "IPv4 retains its synthetic MAC identity",
                input: "10.2.3.4",
                expect: (
                    Some(MacAddress::new([0x02, 0x00, 10, 2, 3, 4])),
                    "02:00:0A:02:03:04".to_string(),
                ),
            },
            Check {
                scenario: "IPv6 uses its full normalized address",
                input: "2001:0db8:0001:0000:0000:0000:0000:0001",
                expect: (None, "ip:2001:db8:1::1".to_string()),
            },
            Check {
                scenario: "IPv6 with the same suffix remains distinct",
                input: "2001:db8:2::1",
                expect: (None, "ip:2001:db8:2::1".to_string()),
            },
        ];
        let inventory = tempfile::NamedTempFile::new().expect("create inventory file");
        let nodes: Vec<_> = cases
            .iter()
            .enumerate()
            .map(|(index, case)| {
                json!({"hostname": format!("host-{index}"), "bmc_ip": case.input, "rack": "rack-a"})
            })
            .collect();
        std::fs::write(
            inventory.path(),
            json!({
                "default_credentials": {"username": "reader", "password": "test-password"},
                "nodes": nodes,
            })
            .to_string(),
        )
        .expect("write inventory file");
        let source = ClusterEndpointSource::from_config(
            ClusterEndpointSourceConfig {
                inventory_path: inventory.path().to_path_buf(),
                port: Some(9443),
                ..Default::default()
            },
            &reqwest(),
            None,
            10,
            None,
        );

        let endpoints = source.fetch_bmc_hosts().await.expect("load inventory");
        assert_eq!(endpoints.len(), cases.len());
        check_values(cases, |ip| {
            let ip: IpAddr = ip.parse().expect("valid inventory IP");
            let endpoint = endpoints
                .iter()
                .find(|endpoint| endpoint.addr.ip == ip)
                .expect("inventory endpoint is retained");
            (endpoint.addr.mac, endpoint.key())
        });

        for endpoint in &endpoints {
            assert_eq!(endpoint.addr.port, Some(9443));
            assert_eq!(endpoint.rack_id, Some(RackId::new("rack-a")));
            assert_eq!(endpoint.log_identity(), endpoint.key());
            assert_eq!(
                EventContext::from_endpoint(endpoint, "test").endpoint_key(),
                endpoint.key(),
            );
            let credentials = endpoint
                .bmc()
                .credential_provider()
                .fetch_credentials(&endpoint.addr)
                .await
                .expect("inventory credentials");
            let BmcCredentials::UsernamePassword { username, password } = credentials else {
                panic!("expected inventory username and password");
            };
            assert_eq!(username, "reader");
            assert_eq!(password.as_deref(), Some("test-password"));
        }
    }

    #[tokio::test]
    async fn file_inventory_keeps_first_row_for_duplicate_bmc_ip() {
        let inventory = tempfile::NamedTempFile::new().expect("create inventory file");
        std::fs::write(
            inventory.path(),
            json!({
                "default_credentials": {"username": "reader", "password": "test-password"},
                "nodes": [
                    {"hostname": "host-a", "bmc_ip": "10.2.3.4", "rack": "rack-a"},
                    {"hostname": "host-b", "bmc_ip": "10.2.3.4", "rack": "rack-b"},
                    {"hostname": "host-c", "bmc_ip": "10.2.3.5", "rack": "rack-b"},
                ],
            })
            .to_string(),
        )
        .expect("write inventory file");
        let source = ClusterEndpointSource::from_config(
            ClusterEndpointSourceConfig {
                inventory_path: inventory.path().to_path_buf(),
                ..Default::default()
            },
            &reqwest(),
            None,
            10,
            None,
        );

        let endpoints = source.fetch_bmc_hosts().await.expect("load inventory");
        let rows: Vec<_> = endpoints
            .iter()
            .map(|endpoint| (endpoint.key(), endpoint.rack_id.clone()))
            .collect();

        assert_eq!(
            rows,
            [
                ("02:00:0A:02:03:04".to_string(), Some(RackId::new("rack-a"))),
                ("02:00:0A:02:03:05".to_string(), Some(RackId::new("rack-b"))),
            ]
        );
    }

    fn file_source(inventory: &tempfile::NamedTempFile) -> ClusterEndpointSource {
        ClusterEndpointSource::from_config(
            ClusterEndpointSourceConfig {
                inventory_path: inventory.path().to_path_buf(),
                port: Some(8443),
                ..Default::default()
            },
            &reqwest(),
            None,
            10,
            None,
        )
    }

    async fn load(inventory: serde_json::Value) -> Result<Vec<Arc<BmcEndpoint>>, String> {
        let file = tempfile::NamedTempFile::new().expect("create inventory file");
        std::fs::write(file.path(), inventory.to_string()).expect("write inventory file");
        let prefix = format!("Invalid cluster inventory {}: ", file.path().display());
        file_source(&file).fetch_bmc_hosts().await.map_err(|e| {
            let message = e.to_string();
            message
                .split_once(&prefix)
                .map_or(message.clone(), |(_, problem)| problem.to_string())
        })
    }

    #[tokio::test]
    async fn file_inventory_collections_carry_identity_and_metadata() {
        let endpoints = load(json!({
            "default_credentials": {"username": "reader", "password": "test-password"},
            "nodes": [{
                "mac": "AA:BB:CC:DD:EE:01", "address": "192.0.2.1", "name": "node-01",
                "rack": "rack-a", "labels": {"site": "dev"},
                "machine": {"serial": "MN-001"}
            }],
            "power_shelves": [{
                "mac": "aa:bb:cc:dd:ee:02", "address": "192.0.2.2", "port": 9443,
                "power_shelf": {"serial": "PS-001"}
            }],
            "switches": [{
                "mac": "aa:bb:cc:dd:ee:03", "address": "192.0.2.3", "switch": {"serial": "SW-001"}
            }],
            "switch_hosts": [{
                "mac": "aa:bb:cc:dd:ee:04", "address": "2001:db8::4", "switch": {"serial": "SW-001"}
            }],
        }))
        .await
        .expect("valid inventory");

        let rows: Vec<_> = endpoints
            .iter()
            .map(|e| {
                let kind = match &e.metadata {
                    Some(EndpointMetadata::Machine(_)) => "machine",
                    Some(EndpointMetadata::PowerShelf(_)) => "power_shelf",
                    Some(EndpointMetadata::Switch(s)) => match s.endpoint_role {
                        crate::endpoint::SwitchEndpointRole::Bmc => "switch_bmc",
                        crate::endpoint::SwitchEndpointRole::Host => "switch_host",
                    },
                    None => "none",
                };
                (e.key(), e.addr.port, kind)
            })
            .collect();
        assert_eq!(
            rows,
            [
                ("AA:BB:CC:DD:EE:01".to_string(), Some(8443), "machine"),
                ("AA:BB:CC:DD:EE:02".to_string(), Some(9443), "power_shelf"),
                ("AA:BB:CC:DD:EE:03".to_string(), Some(8443), "switch_bmc"),
                ("AA:BB:CC:DD:EE:04".to_string(), Some(8443), "switch_host"),
            ]
        );
        assert_eq!(endpoints[0].rack_id, Some(RackId::new("rack-a")));
        assert_eq!(
            endpoints[0].labels.get("site").map(String::as_str),
            Some("dev")
        );
        for endpoint in &endpoints {
            assert_eq!(
                inventory_password(endpoint).await.as_deref(),
                Some("test-password")
            );
        }
    }

    #[tokio::test]
    async fn invalid_file_inventories_are_rejected() {
        let credentials = json!({"username": "reader"});
        let entry = |mac: &str| json!({"mac": mac, "address": "192.0.2.1"});
        let cases = [
            (
                "embedded credentials are required without a credentials file",
                json!({"nodes": [{"hostname": "host-a", "bmc_ip": "10.2.3.4", "rack": "rack-a"}]}),
                Some(
                    "default_credentials is required without endpoint_sources.cluster.credentials_file",
                ),
            ),
            (
                "duplicate MAC across collections, ignoring case",
                json!({
                    "default_credentials": credentials,
                    "nodes": [entry("aa:bb:cc:dd:ee:01")],
                    "switches": [{"mac": "AA:BB:CC:DD:EE:01", "address": "192.0.2.2"}],
                }),
                Some("switches[0].mac duplicates nodes[0].mac"),
            ),
            (
                "metadata for another collection",
                json!({"default_credentials": credentials, "nodes": [
                    {"mac": "aa:bb:cc:dd:ee:01", "address": "192.0.2.1", "power_shelf": {"serial": "PS"}}
                ]}),
                Some("nodes[0] does not support power_shelf metadata"),
            ),
            (
                "switch host without switch metadata",
                json!({"default_credentials": credentials, "switch_hosts": [entry("aa:bb:cc:dd:ee:01")]}),
                Some("switch_hosts[0] requires switch metadata"),
            ),
            (
                "static endpoint validation applies",
                json!({"default_credentials": credentials, "nodes": [
                    {"mac": "aa:bb:cc:dd:ee:01", "address": "192.0.2.1", "labels": {"bmc_ip": "x"}}
                ]}),
                Some("nodes[0].labels key \"bmc_ip\" is reserved"),
            ),
            (
                "per-entry credential profile",
                json!({"default_credentials": credentials, "nodes": [
                    {"mac": "aa:bb:cc:dd:ee:01", "address": "192.0.2.1", "credentials_profile": "bmc"}
                ]}),
                None,
            ),
            (
                "profiles wrapper",
                json!({"default_credentials": credentials, "profiles": {"nodes": [entry("aa:bb:cc:dd:ee:01")]}}),
                None,
            ),
            (
                "switch role in the entry",
                json!({"default_credentials": credentials, "switches": [
                    {"mac": "aa:bb:cc:dd:ee:01", "address": "192.0.2.1", "switch": {"serial": "S", "endpoint_role": "host"}}
                ]}),
                None,
            ),
        ];
        for (scenario, inventory, expected) in cases {
            let error = load(inventory)
                .await
                .err()
                .unwrap_or_else(|| panic!("{scenario}"));
            if let Some(expected) = expected {
                assert_eq!(error, expected, "{scenario}");
            }
        }
    }

    #[tokio::test]
    async fn credentials_file_resolves_collection_profiles_and_rotates() {
        use crate::static_credentials::CredentialFile;

        let credentials = tempfile::NamedTempFile::new().expect("create credentials file");
        let write_credentials = |node_password: &str| {
            std::fs::write(
                credentials.path(),
                json!({
                    "profiles": {
                        "nodes": {"username": "admin", "password": node_password},
                        "switches": {"username": "admin", "password": "switch-secret"},
                    },
                    "overrides": {"aa:bb:cc:dd:ee:03": {"password": "individual"}},
                })
                .to_string(),
            )
            .expect("write credentials file");
        };
        write_credentials("node-secret");
        let inventory = tempfile::NamedTempFile::new().expect("create inventory file");
        let write_inventory = |nodes: serde_json::Value| {
            std::fs::write(
                inventory.path(),
                json!({
                    "default_credentials": {"username": "ignored", "password": "ignored"},
                    "nodes": nodes,
                    "switches": [{"mac": "aa:bb:cc:dd:ee:03", "address": "192.0.2.3"}],
                })
                .to_string(),
            )
            .expect("write inventory file");
        };
        write_inventory(json!([{"mac": "aa:bb:cc:dd:ee:01", "address": "192.0.2.1"}]));
        let file = CredentialFile::open(credentials.path().into())
            .await
            .expect("open credentials file");
        let source = ClusterEndpointSource::from_config_with_request_concurrency(
            ClusterEndpointSourceConfig {
                inventory_path: inventory.path().to_path_buf(),
                credentials_file: Some(credentials.path().into()),
                ..Default::default()
            },
            &reqwest(),
            None,
            10,
            NonZeroUsize::MIN,
            None,
            Some(Arc::new(file)),
        );
        let passwords = || async {
            let mut passwords = Vec::new();
            for endpoint in source.fetch_bmc_hosts().await.expect("load inventory") {
                passwords.push(inventory_password(&endpoint).await);
            }
            passwords
        };

        // The collection selects the profile; a per-MAC override beats it.
        assert_eq!(
            passwords().await,
            [
                Some("node-secret".to_string()),
                Some("individual".to_string())
            ]
        );

        // A rotated profile reaches the running client on the next pass and
        // ends a cooldown the old password caused.
        let first = source.fetch_bmc_hosts().await.expect("load inventory");
        first[0].bmc().reject_credentials_for_test();
        write_credentials("rotated-secret");
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while passwords().await[0].as_deref() != Some("rotated-secret") {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("credentials file reloads");
        let rotated = source.fetch_bmc_hosts().await.expect("load inventory");
        assert!(Arc::ptr_eq(first[0].bmc(), rotated[0].bmc()));
        assert_eq!(first[0].bmc().collector_sweep(), CollectorSweep::Probe);

        // Every entry needs a MAC to select its overrides.
        write_inventory(json!([{"address": "192.0.2.1"}]));
        let error = source
            .fetch_bmc_hosts()
            .await
            .err()
            .expect("missing MAC fails");
        assert!(
            error
                .to_string()
                .ends_with("nodes[0] requires mac with a credentials file"),
            "{error}"
        );
    }

    async fn inventory_password(endpoint: &BmcEndpoint) -> Option<String> {
        let credentials = endpoint
            .bmc()
            .credential_provider()
            .fetch_credentials(&endpoint.addr)
            .await
            .expect("inventory credentials");
        let BmcCredentials::UsernamePassword { password, .. } = credentials else {
            panic!("expected inventory username and password");
        };
        password
    }

    #[tokio::test]
    async fn file_inventory_reuses_clients_and_updates_their_credentials() {
        let inventory = tempfile::NamedTempFile::new().expect("create inventory file");
        let write_inventory = |password: &str, nodes: serde_json::Value| {
            std::fs::write(
                inventory.path(),
                json!({
                    "default_credentials": {"username": "reader", "password": password},
                    "nodes": nodes,
                })
                .to_string(),
            )
            .expect("write inventory file");
        };
        let in_rack =
            |rack: &str| json!([{"hostname": "host-a", "bmc_ip": "10.2.3.4", "rack": rack}]);
        let source = ClusterEndpointSource::from_config(
            ClusterEndpointSourceConfig {
                inventory_path: inventory.path().to_path_buf(),
                ..Default::default()
            },
            &reqwest(),
            None,
            10,
            None,
        );
        let load = || async {
            let endpoints = source.fetch_bmc_hosts().await.expect("load inventory");
            endpoints.into_iter().next()
        };

        write_inventory("old-secret", in_rack("rack-a"));
        let first = load().await.expect("endpoint");
        assert_eq!(
            inventory_password(&first).await.as_deref(),
            Some("old-secret")
        );

        // A rotated password reaches the client running collectors already hold,
        // and ends a cooldown the old password caused.
        first.bmc().reject_credentials_for_test();
        assert_eq!(first.bmc().collector_sweep(), CollectorSweep::Skip);
        write_inventory("new-secret", in_rack("rack-a"));
        let rotated = load().await.expect("endpoint");
        assert!(
            Arc::ptr_eq(first.bmc(), rotated.bmc()),
            "rotation keeps the client"
        );
        assert_eq!(first.bmc().collector_sweep(), CollectorSweep::Probe);
        assert_eq!(
            inventory_password(&first).await.as_deref(),
            Some("new-secret")
        );

        // The client's latency labels name the rack, so a new rack needs a new client.
        write_inventory("new-secret", in_rack("rack-b"));
        let reracked = load().await.expect("endpoint");
        assert!(
            !Arc::ptr_eq(rotated.bmc(), reracked.bmc()),
            "a rack change rebuilds the client"
        );
        assert_eq!(reracked.rack_id, Some(RackId::new("rack-b")));

        // A BMC that leaves the inventory is forgotten; on return it gets a new client.
        write_inventory("new-secret", json!([]));
        assert!(load().await.is_none());
        write_inventory("new-secret", in_rack("rack-b"));
        let returned = load().await.expect("endpoint");
        assert!(
            !Arc::ptr_eq(reracked.bmc(), returned.bmc()),
            "a pruned client is not reused"
        );
    }

    #[tokio::test]
    async fn cluster_manager_failure_keeps_the_last_fetched_credentials() {
        use std::sync::atomic::{AtomicBool, Ordering};

        use axum::Json;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::routing::post;

        let partition_fails = Arc::new(AtomicBool::new(true));
        let fails = partition_fails.clone();
        let app = axum::Router::new().route(
            "/json/",
            post(move |Json(body): Json<serde_json::Value>| {
                let fails = fails.load(Ordering::SeqCst);
                async move {
                    match body["service"].as_str() {
                        Some("cmpart") if fails => StatusCode::BAD_GATEWAY.into_response(),
                        Some("cmpart") => Json(json!({
                            "bmcSettings": {"username": "reader", "password": "manager-secret"}
                        }))
                        .into_response(),
                        _ => Json(json!([
                            {"hostname": "host-a", "bmcIp": "10.2.3.4", "category": "rack-a"}
                        ]))
                        .into_response(),
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        let url = Url::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await });

        let source = ClusterEndpointSource::from_config(
            ClusterEndpointSourceConfig {
                cluster_manager_url: Some(url),
                default_username: "fallback".to_string(),
                default_password: Some("fallback-secret".to_string()),
                ..Default::default()
            },
            &reqwest(),
            None,
            10,
            None,
        );
        let password = || async {
            let endpoints: Vec<Arc<BmcEndpoint>> =
                source.fetch_bmc_hosts().await.expect("load inventory");
            inventory_password(&endpoints[0]).await
        };

        let steps: [(&str, bool, &str); 3] = [
            (
                "nothing fetched yet uses the configured fallback",
                true,
                "fallback-secret",
            ),
            (
                "a successful call installs the partition credentials",
                false,
                "manager-secret",
            ),
            (
                "a failed call keeps the last fetched credentials",
                true,
                "manager-secret",
            ),
        ];
        for (scenario, fails, expected) in steps {
            partition_fails.store(fails, Ordering::SeqCst);
            assert_eq!(password().await.as_deref(), Some(expected), "{scenario}");
        }
        server.abort();
    }
}
