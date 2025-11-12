# Monitoring Integration Guide - Shared Cluster Observability

**Audience**: External projects deploying to the shared Talos Kubernetes cluster
**Purpose**: Leverage centralized Prometheus and Grafana infrastructure for application monitoring
**Last Updated**: 2025-10-23

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Prometheus Integration](#prometheus-integration)
5. [Grafana Integration](#grafana-integration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Overview

The shared Talos Kubernetes cluster provides **centralized monitoring infrastructure** that all projects can leverage:

- **Prometheus**: Metrics collection, storage, and alerting (192.168.200.12:9090)
- **Grafana**: Metrics visualization and dashboards (192.168.200.11:80)
- **AlertManager**: Alert routing and notification (internal)

**Benefits**:
- ✅ Zero infrastructure setup - monitoring ready out-of-the-box
- ✅ Unified observability across all cluster workloads
- ✅ Standardized metrics and dashboards
- ✅ No additional costs or maintenance burden

---

## Architecture

### Monitoring Stack Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Monitoring Stack                      │
│                     (monitoring namespace)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │  Prometheus  │◄─────┤ServiceMonitor├─────►│Your App Pods │ │
│  │192.168.200.12│      │    (CRD)     │      │  /metrics    │ │
│  └──────┬───────┘      └──────────────┘      └──────────────┘ │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │   Grafana    │                                              │
│  │192.168.200.11│                                              │
│  └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Access Points

| Component | LoadBalancer IP | Port | Access | Purpose |
|-----------|-----------------|------|--------|---------|
| **Prometheus** | 192.168.200.12 | 9090 | HTTP | Metrics query API, PromQL |
| **Grafana** | 192.168.200.11 | 80 | HTTP | Dashboard UI |
| **Prometheus API** | 192.168.200.12 | 9090 | HTTP | External metrics push |

**Note**: All LoadBalancers are accessible from the management network (192.168.1.0/24).

---

## Quick Start

### 5-Minute Integration

**Step 1: Expose metrics endpoint in your application**

Your application must expose Prometheus-formatted metrics on `/metrics`:

```go
// Go example
import "github.com/prometheus/client_golang/prometheus/promhttp"

http.Handle("/metrics", promhttp.Handler())
http.ListenAndServe(":8080", nil)
```

```python
# Python example
from prometheus_client import start_http_server, Counter

REQUEST_COUNT = Counter('app_requests_total', 'Total requests')

if __name__ == '__main__':
    start_http_server(8080)  # Exposes /metrics on :8080
```

**Step 2: Create a ServiceMonitor CRD**

```yaml
# your-app-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: your-app
  namespace: your-namespace
  labels:
    app: your-app
spec:
  selector:
    matchLabels:
      app: your-app  # Match your Service labels
  endpoints:
  - port: metrics  # Service port name
    interval: 30s  # Scrape every 30 seconds
    path: /metrics # Metrics endpoint path
```

**Step 3: Deploy and verify**

```bash
kubectl apply -f your-app-servicemonitor.yaml

# Verify Prometheus discovers your ServiceMonitor
curl http://192.168.200.12:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "your-app")'
```

**Done!** Prometheus is now scraping your application metrics.

---

## Prometheus Integration

### ServiceMonitor CRD

**ServiceMonitor** is a Custom Resource Definition (CRD) that tells Prometheus how to discover and scrape your application's metrics.

#### Basic ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp
  namespace: myapp-prod
  labels:
    app: myapp
    team: backend
spec:
  # Select Services to monitor
  selector:
    matchLabels:
      app: myapp
      component: api

  # Define scrape endpoints
  endpoints:
  - port: metrics          # Service port name (not number)
    interval: 30s          # Scrape frequency
    path: /metrics         # Metrics path
    scheme: http           # http or https

    # Optional: Relabel metrics
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_name]
      targetLabel: pod
      action: replace
```

#### Multi-Port ServiceMonitor

If your application exposes multiple metrics endpoints:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp-multi
  namespace: myapp-prod
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  # Application metrics
  - port: app-metrics
    interval: 30s
    path: /metrics

  # Database metrics
  - port: db-metrics
    interval: 60s
    path: /db/metrics

  # Custom metrics
  - port: custom
    interval: 15s
    path: /custom-metrics
```

#### HTTPS/TLS ServiceMonitor

For applications with TLS-enabled metrics endpoints:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp-tls
  namespace: myapp-prod
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    scheme: https
    tlsConfig:
      # Skip TLS verification (not recommended for production)
      insecureSkipVerify: true

      # OR: Use CA certificate
      ca:
        secret:
          name: myapp-ca-cert
          key: ca.crt
```

#### Authentication

For metrics endpoints requiring authentication:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp-auth
  namespace: myapp-prod
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics

    # Basic Auth
    basicAuth:
      username:
        name: myapp-metrics-auth
        key: username
      password:
        name: myapp-metrics-auth
        key: password
```

### Service Configuration Requirements

Your Kubernetes Service **must** have a named port for ServiceMonitor to work:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: myapp-prod
  labels:
    app: myapp  # Must match ServiceMonitor selector
spec:
  selector:
    app: myapp
  ports:
  - name: metrics  # REQUIRED: Named port
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: http
    port: 80
    targetPort: 8080
```

### Verification

**Check ServiceMonitor is created:**
```bash
kubectl get servicemonitor -n your-namespace
```

**Verify Prometheus discovered your targets:**
```bash
# Via API
curl http://192.168.200.12:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "your-app")'

# Via Prometheus UI
# Navigate to: http://192.168.200.12:9090/targets
# Search for your application name
```

**Test PromQL query:**
```bash
# Check if metrics are being scraped
curl 'http://192.168.200.12:9090/api/v1/query?query=up{job="your-app"}'
```

---

## Grafana Integration

### Accessing Grafana

**URL**: http://192.168.200.11
**Authentication**: Contact cluster administrator for credentials

### Creating Dashboards

#### Option 1: Grafana UI (Recommended for Development)

1. Navigate to http://192.168.200.11
2. Login with provided credentials
3. Click **+ → Create Dashboard**
4. Add panels and configure queries
5. Save dashboard

#### Option 2: Dashboard as Code (Recommended for Production)

Deploy dashboards as Kubernetes ConfigMaps for GitOps workflows:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-dashboard
  namespace: monitoring  # MUST be monitoring namespace
  labels:
    grafana_dashboard: "1"  # REQUIRED: Auto-discovery label
data:
  myapp-dashboard.json: |
    {
      "dashboard": {
        "title": "MyApp Metrics",
        "panels": [
          {
            "title": "Request Rate",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"myapp\"}[5m])",
                "legendFormat": "{{method}} {{path}}"
              }
            ],
            "type": "graph"
          }
        ],
        "uid": "myapp-metrics",
        "version": 1
      },
      "overwrite": true
    }
```

**Deploy:**
```bash
kubectl apply -f myapp-dashboard-configmap.yaml
```

Grafana will automatically discover and import the dashboard within 60 seconds.

### Dashboard Best Practices

**1. Use consistent naming:**
```
Format: [Project] - [Component] - [Metric Type]
Example: "OrgCash - API - Performance Metrics"
```

**2. Include essential panels:**
- Request rate (RPS)
- Error rate (4xx/5xx)
- Latency (p50, p95, p99)
- Resource usage (CPU, memory)

**3. Use templating for flexibility:**
```json
{
  "templating": {
    "list": [
      {
        "name": "namespace",
        "type": "query",
        "query": "label_values(up, namespace)"
      },
      {
        "name": "pod",
        "type": "query",
        "query": "label_values(up{namespace=\"$namespace\"}, pod)"
      }
    ]
  }
}
```

### Common PromQL Queries

**Request rate (per second):**
```promql
rate(http_requests_total{job="myapp"}[5m])
```

**Error rate (percentage):**
```promql
sum(rate(http_requests_total{job="myapp",status=~"5.."}[5m]))
/
sum(rate(http_requests_total{job="myapp"}[5m])) * 100
```

**p95 latency:**
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="myapp"}[5m]))
```

**Memory usage:**
```promql
container_memory_usage_bytes{pod=~"myapp-.*"}
```

**CPU usage (cores):**
```promql
rate(container_cpu_usage_seconds_total{pod=~"myapp-.*"}[5m])
```

---

## Best Practices

### Metric Naming

Follow Prometheus naming conventions:

**Format**: `{namespace}_{subsystem}_{name}_{unit}`

**Examples:**
```
✅ myapp_http_requests_total
✅ myapp_database_query_duration_seconds
✅ myapp_cache_hits_total
✅ myapp_queue_messages_in_flight

❌ MyApp.HTTP.Requests (wrong format)
❌ requests (too generic)
❌ app_request_count_total (redundant "count" and "total")
```

### Metric Types

**Counter** - Monotonically increasing value:
```go
requestsTotal := prometheus.NewCounter(prometheus.CounterOpts{
    Name: "myapp_http_requests_total",
    Help: "Total HTTP requests",
})
```

**Gauge** - Value that can go up or down:
```go
activeConnections := prometheus.NewGauge(prometheus.GaugeOpts{
    Name: "myapp_active_connections",
    Help: "Number of active connections",
})
```

**Histogram** - Distribution of values (latencies):
```go
requestDuration := prometheus.NewHistogram(prometheus.HistogramOpts{
    Name: "myapp_http_request_duration_seconds",
    Help: "HTTP request duration in seconds",
    Buckets: []float64{0.1, 0.5, 1.0, 2.5, 5.0, 10.0},
})
```

**Summary** - Similar to histogram but calculates quantiles:
```go
requestSize := prometheus.NewSummary(prometheus.SummaryOpts{
    Name: "myapp_http_request_size_bytes",
    Help: "HTTP request size in bytes",
    Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
})
```

### Labels

**Use labels for dimensions:**
```go
httpRequests := prometheus.NewCounterVec(
    prometheus.CounterOpts{
        Name: "myapp_http_requests_total",
        Help: "Total HTTP requests",
    },
    []string{"method", "path", "status"},
)

// Usage
httpRequests.WithLabelValues("GET", "/api/users", "200").Inc()
```

**⚠️ Cardinality Warning**: Avoid high-cardinality labels (user IDs, IP addresses, timestamps)
```
❌ {user_id="12345"}  # Millions of unique values
❌ {ip_address="192.168.1.100"}  # Too many unique IPs
✅ {status_code="200"}  # Limited set of values
✅ {method="GET"}  # Limited set of values
```

### Scrape Intervals

**Recommended intervals:**
- **Critical services**: 15s
- **Standard applications**: 30s (default)
- **Background jobs**: 60s
- **Infrequent metrics**: 120s

**Trade-offs:**
- Lower interval = higher resolution, more storage, more load
- Higher interval = lower resolution, less storage, less load

### Resource Limits

**Metrics endpoint should be lightweight:**
- Keep scrape duration < 1 second
- Limit metric cardinality (< 1000 unique series per service)
- Use sampling for high-frequency events

---

## Troubleshooting

### Metrics Not Appearing in Prometheus

**1. Check ServiceMonitor exists:**
```bash
kubectl get servicemonitor -n your-namespace your-app
```

**2. Verify Service has correct labels:**
```bash
kubectl get svc -n your-namespace your-app -o yaml | grep -A 5 "labels:"
```

**3. Check Prometheus targets:**
```bash
# Via UI
http://192.168.200.12:9090/targets

# Via API
curl http://192.168.200.12:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "your-app")'
```

**4. Verify metrics endpoint is accessible:**
```bash
kubectl port-forward -n your-namespace svc/your-app 8080:8080
curl http://localhost:8080/metrics
```

**5. Check ServiceMonitor configuration:**
```bash
kubectl describe servicemonitor -n your-namespace your-app
```

### Common Errors

**Error: "No targets found"**
- **Cause**: ServiceMonitor selector doesn't match Service labels
- **Fix**: Ensure `selector.matchLabels` matches Service `metadata.labels`

**Error: "Connection refused"**
- **Cause**: Incorrect port name or port not exposed
- **Fix**: Verify Service has named port matching ServiceMonitor `endpoints.port`

**Error: "Metrics endpoint returns 404"**
- **Cause**: Incorrect `path` in ServiceMonitor
- **Fix**: Verify application exposes metrics on specified path (default: `/metrics`)

**Error: "High cardinality metrics"**
- **Cause**: Too many unique label values
- **Fix**: Remove high-cardinality labels (user IDs, timestamps, IPs)

### Dashboard Not Appearing in Grafana

**1. Verify ConfigMap in monitoring namespace:**
```bash
kubectl get configmap -n monitoring your-dashboard
```

**2. Check ConfigMap has required label:**
```bash
kubectl get configmap -n monitoring your-dashboard -o yaml | grep grafana_dashboard
```

**3. Wait for Grafana sidecar to sync (60 seconds)**

**4. Check Grafana logs:**
```bash
kubectl logs -n monitoring -l app.kubernetes.io/name=grafana -c grafana-sc-dashboard
```

---

## Examples

### Example 1: Go Application with Prometheus Metrics

**main.go:**
```go
package main

import (
    "net/http"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    httpRequests = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "myapp_http_requests_total",
            Help: "Total HTTP requests by method and path",
        },
        []string{"method", "path", "status"},
    )

    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "myapp_http_request_duration_seconds",
            Help: "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "path"},
    )
)

func init() {
    prometheus.MustRegister(httpRequests)
    prometheus.MustRegister(requestDuration)
}

func main() {
    http.HandleFunc("/api/users", func(w http.ResponseWriter, r *http.Request) {
        timer := prometheus.NewTimer(requestDuration.WithLabelValues(r.Method, r.URL.Path))
        defer timer.ObserveDuration()

        // Your handler logic here

        httpRequests.WithLabelValues(r.Method, r.URL.Path, "200").Inc()
        w.WriteHeader(http.StatusOK)
    })

    // Expose metrics endpoint
    http.Handle("/metrics", promhttp.Handler())

    http.ListenAndServe(":8080", nil)
}
```

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: myapp-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - name: http
          containerPort: 8080
        - name: metrics
          containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: myapp-prod
  labels:
    app: myapp
spec:
  selector:
    app: myapp
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 8080
    targetPort: 8080
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp
  namespace: myapp-prod
  labels:
    app: myapp
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Example 2: Python Flask Application

**app.py:**
```python
from flask import Flask, request
from prometheus_client import Counter, Histogram, generate_latest
import time

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'myapp_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'myapp_http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

@app.route('/api/data')
def get_data():
    start = time.time()

    # Your handler logic here
    result = {"data": "example"}

    duration = time.time() - start
    REQUEST_LATENCY.labels(method='GET', endpoint='/api/data').observe(duration)
    REQUEST_COUNT.labels(method='GET', endpoint='/api/data', status='200').inc()

    return result

@app.route('/metrics')
def metrics():
    return generate_latest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Example 3: Node.js Express Application

**server.js:**
```javascript
const express = require('express');
const client = require('prom-client');

const app = express();

// Create a Registry
const register = new client.Registry();

// Add default metrics
client.collectDefaultMetrics({ register });

// Custom metrics
const httpRequestCounter = new client.Counter({
  name: 'myapp_http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'path', 'status'],
  registers: [register]
});

const httpRequestDuration = new client.Histogram({
  name: 'myapp_http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'path'],
  registers: [register]
});

// Middleware to track metrics
app.use((req, res, next) => {
  const end = httpRequestDuration.startTimer({ method: req.method, path: req.path });

  res.on('finish', () => {
    end();
    httpRequestCounter.inc({ method: req.method, path: req.path, status: res.statusCode });
  });

  next();
});

// Your routes
app.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

app.listen(8080, () => {
  console.log('Server listening on port 8080');
});
```

### Example 4: Complete Dashboard ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  myapp-dashboard.json: |
    {
      "dashboard": {
        "title": "MyApp Performance Metrics",
        "tags": ["myapp", "performance"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Request Rate (RPS)",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
              {
                "expr": "rate(myapp_http_requests_total{namespace=\"myapp-prod\"}[5m])",
                "legendFormat": "{{method}} {{path}}"
              }
            ]
          },
          {
            "id": 2,
            "title": "Error Rate (%)",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "targets": [
              {
                "expr": "sum(rate(myapp_http_requests_total{namespace=\"myapp-prod\",status=~\"5..\"}[5m])) / sum(rate(myapp_http_requests_total{namespace=\"myapp-prod\"}[5m])) * 100",
                "legendFormat": "Error Rate"
              }
            ]
          },
          {
            "id": 3,
            "title": "Request Latency (p95)",
            "type": "graph",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(myapp_http_request_duration_seconds_bucket{namespace=\"myapp-prod\"}[5m]))",
                "legendFormat": "p95 latency"
              }
            ]
          }
        ],
        "uid": "myapp-perf",
        "version": 1
      },
      "overwrite": true
    }
```

---

## Support

**Questions or Issues?**
- Check Prometheus targets: http://192.168.200.12:9090/targets
- Review Prometheus logs: `kubectl logs -n monitoring prometheus-prometheus-prometheus-0`
- Review Grafana logs: `kubectl logs -n monitoring -l app.kubernetes.io/name=grafana`
- Contact cluster administrator: thomas@example.com

**Useful Links:**
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Client Libraries](https://prometheus.io/docs/instrumenting/clientlibs/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)

---

**Document Version**: 1.0
**Created**: 2025-10-23
**Cluster**: Talos Homelab (admin@homelab-test)
**Maintained By**: Platform Team
